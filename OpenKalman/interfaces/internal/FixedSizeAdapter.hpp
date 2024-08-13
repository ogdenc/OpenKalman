/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Interfaces for FixedSizeAdapter.
 */

#ifndef OPENKALMAN_INTERFACES_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_INTERFACES_FIXEDSIZEADAPTER_HPP


namespace OpenKalman::interface
{
  // --------------------------- //
  //   indexible_object_traits   //
  // --------------------------- //

  template<typename NestedMatrix, typename...Ds>
  struct indexible_object_traits<internal::FixedSizeAdapter<NestedMatrix, Ds...>>
  {
  private:

    using Xpr = internal::FixedSizeAdapter<NestedMatrix, Ds...>;


    // Truncate any trailing ℝ¹ dimensions
    template<std::size_t N = sizeof...(Ds)>
    static constexpr auto count_indices_impl()
    {
      if constexpr (N == 0)
        return N;
      else if constexpr (equivalent_to<Dimensions<1>, std::tuple_element_t<N - 1, std::tuple<Ds...>>>)
        return count_indices_impl<N - 1>();
      else
        return N;
    }

  public:

    using scalar_type = scalar_type_of_t<NestedMatrix>;


    template<typename Arg>
    static constexpr auto count_indices(const Arg&)
    {
      return std::integral_constant<std::size_t, count_indices_impl()>{};
    }


    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, const N& n)
    {
      if constexpr (sizeof...(Ds) > 0 and static_index_value<N>)
      {
        using D = std::tuple_element_t<N::value, std::tuple<Ds...>>;
        if constexpr (fixed_vector_space_descriptor<D>) return std::decay_t<D> {};
        else return OpenKalman::get_vector_space_descriptor(OpenKalman::nested_object(arg), n);
      }
      else if constexpr (equivalent_to<Dimensions<1>, Ds...>)
      {
        return Dimensions<1>{};
      }
      else
      {
        return OpenKalman::get_vector_space_descriptor(OpenKalman::nested_object(arg), n);
      }
    }


    using dependents = std::tuple<NestedMatrix>;


    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nested_object();
    }


#ifdef __cpp_concepts
    template<typename Arg> requires constant_matrix<NestedMatrix> or (... and equivalent_to<Ds, Dimensions<1>>)
#else
    template<typename M = NestedMatrix, typename Arg, std::enable_if_t<
      constant_matrix<M> or (... and equivalent_to<Ds, Dimensions<1>>), int> = 0>
#endif
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<NestedMatrix>) return constant_coefficient {OpenKalman::nested_object(arg)};
      else return internal::get_singular_component(arg);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires constant_diagonal_matrix<NestedMatrix>
#else
    template<typename M = NestedMatrix, typename Arg, std::enable_if_t<constant_diagonal_matrix<M>, int> = 0>
#endif
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {OpenKalman::nested_object(arg)};
    }


    // one_dimensional is not necessary


    // is_square is not necessary


    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<NestedMatrix, Qualification::depends_on_dynamic_shape>;


    static constexpr bool is_writable = writable<NestedMatrix>;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires raw_data_defined_for<NestedMatrix>
#else
    template<typename N = NestedMatrix, typename Arg, std::enable_if_t<raw_data_defined_for<N>, int> = 0>
#endif
    static constexpr auto * const
    raw_data(Arg& arg)
    {
      return internal::raw_data(OpenKalman::nested_object(arg));
    }


    static constexpr Layout layout = layout_of_v<NestedMatrix>;


#ifdef __cpp_concepts
    template<typename Arg> requires (layout == Layout::stride)
#else
    template<Layout l = layout, typename Arg, std::enable_if_t<l == Layout::stride, int> = 0>
#endif
    static auto
    strides(Arg&& arg)
    {
      return OpenKalman::internal::strides(OpenKalman::nested_object(std::forward<Arg>(arg)));
    }

  };


  // ------------------- //
  //  library_interface  //
  // ------------------- //

  template<typename NestedMatrix, typename...Ds>
  struct library_interface<internal::FixedSizeAdapter<NestedMatrix, Ds...>> : library_interface<std::decay_t<NestedMatrix>>
  {
  private:

    using Nested = std::decay_t<NestedMatrix>;
    using NestedInterface = library_interface<Nested>;

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, NestedMatrix>;

  private:

    template<std::size_t N, typename Indices>
    static constexpr decltype(auto) add_trailing_indices(const Indices& indices)
    {
      constexpr auto M = static_range_size_v<Indices>;
      if constexpr (M != dynamic_size and N != dynamic_size and N > M)
      {
        std::array<std::size_t, N> ret;
        std::fill(std::copy(indices.begin(), indices.end(), ret.begin()), ret.end(), 0);
        return ret;
      }
      else return indices;
    }

  public:

#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>>
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      return NestedInterface::get_component(nested_object(std::forward<Arg>(arg)), add_trailing_indices<index_count_v<Nested>>(indices));
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      interface::set_component_defined_for<Nested, decltype(std::declval<Arg&>().nested_object()), const scalar_type_of_t<Arg>&, std::initializer_list<std::size_t>>
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::set_component_defined_for<Nested, typename nested_object_of<Arg&>::type, const typename scalar_type_of<Arg>::type&, std::initializer_list<std::size_t>>, int> = 0>
#endif
    static constexpr void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      NestedInterface::set_component(OpenKalman::nested_object(arg), s, add_trailing_indices<index_count_v<Nested>>(indices));
    }


#ifdef __cpp_concepts
    template<typename A> requires
      interface::to_native_matrix_defined_for<Nested, A&&> or
      interface::to_native_matrix_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<A&&>))>
#else
    template<typename A, std::enable_if_t<
      interface::to_native_matrix_defined_for<Nested, A&&> or
      interface::to_native_matrix_defined_for<Nested, typename nested_object_of<A&&>::type>, int> = 0>
#endif
    static decltype(auto)
    to_native_matrix(A&& a)
    {
      if constexpr (interface::to_native_matrix_defined_for<Nested, A&&>)
        return NestedInterface::to_native_matrix(std::forward<A>(a));
      else
        return NestedInterface::to_native_matrix(OpenKalman::nested_object(std::forward<A>(a)));
    }


    // make_default is inherited

    // fill_components is inherited

    // make_constant is inherited

    // make_identity_matrix is inherited


#ifdef __cpp_concepts
    template<TriangleType t, indexible Arg> requires
      interface::make_triangular_matrix_defined_for<Nested, t, Arg&&> or
      interface::make_triangular_matrix_defined_for<Nested, t, decltype(OpenKalman::nested_object(std::declval<Arg&&>()))>
    static constexpr triangular_matrix<t> auto
#else
    template<TriangleType t, typename Arg, std::enable_if_t<
      interface::make_triangular_matrix_defined_for<Nested, t, Arg&&> or
      interface::make_triangular_matrix_defined_for<Nested, t, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    make_triangular_matrix(Arg&& arg)
    {
      if constexpr (interface::make_triangular_matrix_defined_for<Nested, t, Arg&&>)
      {
        return NestedInterface::template make_triangular_matrix<t>(std::forward<Arg>(arg));
      }
      else
      {
        auto tri = NestedInterface::template make_triangular_matrix<t>(OpenKalman::nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType t, indexible Arg> requires
      interface::make_hermitian_adapter_defined_for<Nested, t, Arg&&> or
      interface::make_hermitian_adapter_defined_for<Nested, t, decltype(OpenKalman::nested_object(std::declval<Arg&&>()))>
    static constexpr hermitian_matrix auto
#else
    template<HermitianAdapterType t, typename Arg, std::enable_if_t<
      make_hermitian_adapter_defined_for<Nested, t, Arg&> or
      make_hermitian_adapter_defined_for<Nested, t, typename nested_object_of<Arg&>::type>, int> = 0>
    static constexpr auto
#endif
    make_hermitian_adapter(Arg&& arg)
    {
      if constexpr (interface::make_hermitian_adapter_defined_for<Nested, t, Arg&&>)
      {
        return NestedInterface::template hermitian_adapter<t>(std::forward<Arg>(arg));
      }
      else
      {
        auto h = NestedInterface::template make_hermitian_adapter<t>(OpenKalman::nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_square_adapter_like(std::move(h));
      }
    }


    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto)
    get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      return NestedInterface::get_block(OpenKalman::nested_object(std::forward<Arg>(arg)), begin, size);
    };


    template<typename Arg, typename Block, typename...Begin>
    static Arg&
    set_block(Arg& arg, Block&& block, const Begin&...begin)
    {
      NestedInterface::set_block(OpenKalman::nested_object(arg), std::forward<Block>(block), begin...);
      return arg;
    };


#ifdef __cpp_concepts
    template<TriangleType t, typename A, typename B> requires
      interface::set_triangle_defined_for<Nested, t, A&&, B&&> or
      interface::set_triangle_defined_for<Nested, t, decltype(OpenKalman::nested_object(std::declval<A&&>)), B&&>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      interface::set_triangle_defined_for<Nested, t, A&&, B&&> or
      interface::set_triangle_defined_for<Nested, t, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static decltype(auto)
    set_triangle(A&& a, B&& b)
    {
      if constexpr (interface::set_triangle_defined_for<Nested, t, A&&, B&&>)
      {
        return NestedInterface::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        auto&& ret = NestedInterface::template set_triangle<t>(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b));
        using Ret = decltype(ret);
        if constexpr (std::is_lvalue_reference_v<Ret> and std::is_same_v<Ret, decltype(OpenKalman::nested_object(std::declval<A&&>()))>)
          return std::forward<A>(a);
        else
        {
          using D0 = vector_space_descriptor_of_t<A, 0>;
          using D1 = vector_space_descriptor_of_t<A, 1>;
          auto r = internal::make_fixed_square_adapter_like<D0, D1>(std::forward<Ret>(ret));
          return r;
        }
      }
    }


#ifdef __cpp_concepts
    template<vector<0, Qualification::depends_on_dynamic_shape> Arg> requires
      interface::to_diagonal_defined_for<Nested, Arg&&> or
      interface::to_diagonal_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>))>
    static constexpr diagonal_matrix auto
    to_diagonal(Arg&& arg)
#else
    template<typename Arg, std::enable_if_t<
      interface::to_diagonal_defined_for<Nested, Arg&&> or
      interface::to_diagonal_defined_for<Nested, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
    to_diagonal(Arg&& arg)
#endif
    {
      if constexpr (one_dimensional<Arg>) return std::forward<Arg>(arg);
      else
      {
        if constexpr (interface::to_diagonal_defined_for<Nested, Arg&&>)
        {
          return NestedInterface::to_diagonal(std::forward<Arg>(arg));
        }
        else
        {
          auto diag = NestedInterface::to_diagonal(OpenKalman::nested_object(std::forward<Arg>(arg)));
          return internal::make_fixed_square_adapter_like<vector_space_descriptor_of_t<Arg, 0>>(std::move(diag));
        }
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg>
    static constexpr vector auto
#else
    template<typename Arg>
    static constexpr auto
#endif
    diagonal_of(Arg&& arg)
    {
      if constexpr (one_dimensional<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else if constexpr (interface::diagonal_of_defined_for<NestedInterface, Arg&&>)
      {
        return NestedInterface::diagonal_of(std::forward<Arg>(arg));
      }
      else
      {
        using D = decltype(get_vector_space_descriptor(arg, internal::smallest_dimension_index(arg)));
        return internal::make_fixed_size_adapter<D>(NestedInterface::diagonal_of(OpenKalman::nested_object(std::forward<Arg>(arg))));
      }
    }

  private:

    template<std::size_t Ix, typename Arg, typename Factors_tup>
    static constexpr auto broadcast_for_index(const Arg& arg, const Factors_tup& factors_tup)
    {
      constexpr auto N = std::tuple_size_v<Factors_tup>;
      if constexpr (Ix < N)
        return replicate_vector_space_descriptor(get_vector_space_descriptor<Ix>(arg), std::get<Ix>(factors_tup));
      else
        return Dimensions<1>{};
    }


    template<typename Arg, std::size_t...Is, typename Factors_tup>
    static constexpr auto broadcast_impl(Arg&& arg, std::index_sequence<Is...>, const Factors_tup& factors_tup)
    {
      constexpr auto N = std::tuple_size_v<Factors_tup>;
      return internal::make_fixed_size_adapter<decltype(broadcast_for_index<Is>(arg, factors_tup))...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg, index_value...Factors> requires
      interface::broadcast_defined_for<Nested, Arg&&, const Factors&...> or
      interface::broadcast_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), const Factors&...>
    static indexible auto
#else
    template<typename Arg, typename...Factors, std::enable_if_t<
      interface::broadcast_defined_for<Nested, Arg&&, const Factors&...> or
      interface::broadcast_defined_for<Nested, typename nested_object_of<Arg&&>::type, const Factors&...>, int> = 0>
    static auto
#endif
    broadcast(Arg&& arg, const Factors&...factors)
    {
      if constexpr (interface::broadcast_defined_for<Nested, Arg&&, const Factors&...>)
      {
        return NestedInterface::broadcast(std::forward<Arg>(arg), factors...);
      }
      else
      {
        auto&& ret = NestedInterface::broadcast(OpenKalman::nested_object(std::forward<Arg>(arg)), factors...);
        using Ret = decltype(ret);
        auto seq = std::make_index_sequence<std::max(index_count_v<Arg>, sizeof...(factors))>{};
        return broadcast_impl(std::forward<Ret>(ret), seq, std::forward_as_tuple(factors...));
      }
    }


#ifdef __cpp_concepts
    template<vector_space_descriptor...IDs, typename Operation, indexible...Args> requires
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<IDs...>&, Operation&&>
    static indexible auto
#else
    template<typename...IDs, typename Operation, typename...Args, std::enable_if_t<
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<IDs...>&, Operation&&>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<IDs...>& d_tup, Operation&& op)
    {
      return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op));
    }


#ifdef __cpp_concepts
    template<vector_space_descriptor...IDs, typename Operation, indexible Arg, indexible...Args> requires
      interface::n_ary_operation_defined_for<Nested, const std::tuple<IDs...>&, Operation&&, Arg, Args...> or
      interface::n_ary_operation_defined_for<Nested, const std::tuple<IDs...>&, Operation&&, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), Args...>
    static indexible auto
#else
    template<typename...IDs, typename Operation, typename Arg, typename...Args, std::enable_if_t<
      interface::n_ary_operation_defined_for<Nested, const std::tuple<IDs...>&, Operation&&, Arg, Args...> or
      interface::n_ary_operation_defined_for<Nested, const std::tuple<IDs...>&, Operation&&, typename nested_object_of<Arg&&>::type, Args...>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<IDs...>& d_tup, Operation&& op, Arg&& arg, Args&&...args)
    {
      if constexpr (interface::n_ary_operation_defined_for<Nested, const std::tuple<IDs...>&, Operation&&, Arg, Args...>)
      {
        return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op), std::forward<Arg>(arg), std::forward<Args>(args)...);
      }
      else
      {
        return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op), OpenKalman::nested_object(std::forward<Arg>(arg)), std::forward<Args>(args)...);
      }
    }


#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, indexible Arg> requires
      interface::reduce_defined_for<Nested, BinaryFunction&&, Arg&&, indices...> or
      interface::reduce_defined_for<Nested, BinaryFunction&&, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), indices...>
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
      interface::reduce_defined_for<Nested, BinaryFunction&&, Arg&&, indices...> or
      interface::reduce_defined_for<Nested, BinaryFunction&&, typename nested_object_of<Arg&&>::type, indices...>, int> = 0>
#endif
    static constexpr auto
    reduce(BinaryFunction&& op, Arg&& arg)
    {
      if constexpr (interface::reduce_defined_for<Nested, BinaryFunction&&, Arg&&, indices...>)
        return NestedInterface::template reduce<indices...>(std::forward<BinaryFunction>(op), std::forward<Arg>(arg));
      else
        return NestedInterface::template reduce<indices...>(std::forward<BinaryFunction>(op), OpenKalman::nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C> requires
      interface::to_euclidean_defined_for<Nested, Arg&&, const C&> or
      interface::to_euclidean_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), const C&>
    static constexpr indexible auto
#else
    template<typename Arg, typename C, std::enable_if_t<
      interface::to_euclidean_defined_for<Nested, Arg&&, const C&> or
      interface::to_euclidean_defined_for<Nested, typename nested_object_of<Arg&&>::type, const C&>, int> = 0>
    static constexpr auto
#endif
    to_euclidean(Arg&& arg, const C& c)
    {
      if constexpr (interface::to_euclidean_defined_for<Nested, Arg&&, const C&>)
        return NestedInterface::to_euclidean(std::forward<Arg>(arg), c);
      else
        return NestedInterface::to_euclidean(OpenKalman::nested_object(std::forward<Arg>(arg)), c);
    }


#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C> requires
      interface::from_euclidean_defined_for<Nested, Arg&&, const C&> or
      interface::from_euclidean_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), const C&>
    static constexpr indexible auto
#else
    template<typename Arg, typename C, std::enable_if_t<
      interface::from_euclidean_defined_for<Nested, Arg&&, const C&> or
      interface::from_euclidean_defined_for<Nested, typename nested_object_of<Arg&&>::type, const C&>, int> = 0>
    static constexpr auto
#endif
    from_euclidean(Arg&& arg, const C& c)
    {
      if constexpr (interface::from_euclidean_defined_for<Nested, Arg&&, const C&>)
        return NestedInterface::from_euclidean(std::forward<Arg>(arg), c);
      else
        return NestedInterface::from_euclidean(OpenKalman::nested_object(std::forward<Arg>(arg)), c);
    }


#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C> requires
      interface::wrap_angles_defined_for<Nested, Arg&&, const C&> or
      interface::wrap_angles_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), const C&>
    static constexpr indexible auto
#else
    template<typename Arg, typename C, std::enable_if_t<
      interface::wrap_angles_defined_for<Nested, Arg&&, const C&> or
      interface::wrap_angles_defined_for<Nested, typename nested_object_of<Arg&&>::type, const C&>, int> = 0>
    static constexpr auto
#endif
    wrap_angles(Arg&& arg, const C& c)
    {
      if constexpr (interface::wrap_angles_defined_for<Nested, Arg&&, const C&>)
        return NestedInterface::wrap_angles(std::forward<Arg>(arg), c);
      else
        return NestedInterface::wrap_angles(OpenKalman::nested_object(std::forward<Arg>(arg)), c);
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::conjugate_defined_for<Nested, Arg&&> or
      interface::conjugate_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>))>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::conjugate_defined_for<Nested, Arg&&> or
      interface::conjugate_defined_for<Nested, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    conjugate(Arg&& arg)
    {
      if constexpr (interface::conjugate_defined_for<Nested, Arg&&>)
      {
        return NestedInterface::conjugate(std::forward<Arg>(arg));
      }
      else
      {
        auto&& conj = NestedInterface::conjugate(OpenKalman::nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_size_adapter_like<Arg>(std::forward<decltype(conj)>(conj));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::transpose_defined_for<Nested, Arg&&> or
      interface::transpose_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>))>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::transpose_defined_for<Nested, Arg&&> or
      interface::transpose_defined_for<Nested, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    transpose(Arg&& arg)
    {
      if constexpr (interface::transpose_defined_for<Nested, Arg&&>)
      {
        return NestedInterface::transpose(std::forward<Arg>(arg));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>>(
          NestedInterface::transpose(OpenKalman::nested_object(std::forward<Arg>(arg))));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::adjoint_defined_for<Nested, Arg&&> or
      interface::adjoint_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>))>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::adjoint_defined_for<Nested, Arg&&> or
      interface::adjoint_defined_for<Nested, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    adjoint(Arg&& arg)
    {
      if constexpr (interface::adjoint_defined_for<Nested, Arg&&>)
      {
        return NestedInterface::adjoint(std::forward<Arg>(arg));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>>(
          NestedInterface::adjoint(OpenKalman::nested_object(std::forward<Arg>(arg))));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::determinant_defined_for<Nested, Arg&&> or
      interface::determinant_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>))>
    static constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
    template<typename Arg, std::enable_if_t<
      interface::determinant_defined_for<Nested, Arg&&> or
      interface::determinant_defined_for<Nested, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    determinant(Arg&& arg)
    {
      if constexpr (interface::determinant_defined_for<Nested, Arg&&>)
      {
        return NestedInterface::determinant(std::forward<Arg>(arg));
      }
      else
      {
        return NestedInterface::determinant(OpenKalman::nested_object(std::forward<Arg>(arg)));
      }
    }


#ifdef __cpp_concepts
    template<typename Arg, typename...Args> requires
      interface::sum_defined_for<Nested, Arg&&, Args&&...> or
      interface::sum_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<Arg&&>)), Args&&...>
#else
    template<typename Arg, typename...Args, std::enable_if_t<
      interface::sum_defined_for<Nested, Arg&&, Args&&...> or
      interface::sum_defined_for<Nested, typename nested_object_of<Arg&&>::type, Args&&...>, int> = 0>
#endif
    static auto
    sum(Arg&& arg, Args&&...args)
    {
      if constexpr (interface::sum_defined_for<Nested, Arg&&, Args&&...>)
      {
        return NestedInterface::sum(std::forward<Arg>(arg), std::forward<Args>(args)...);
      }
      else
      {
        return NestedInterface::sum(OpenKalman::nested_object(std::forward<Arg>(arg)), std::forward<Args>(args)...);
      }
    }


#ifdef __cpp_concepts
    template<typename A, typename B> requires
      interface::contract_defined_for<Nested, A&&, B&&> or
      interface::contract_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<A&&>)), B&&>
#else
    template<typename A, typename B, std::enable_if_t<
      interface::contract_defined_for<Nested, A&&, B&&> or
      interface::contract_defined_for<Nested, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static auto
    contract(A&& a, B&& b)
    {
      if constexpr (interface::contract_defined_for<Nested, A&&, B&&>)
      {
        return NestedInterface::contract(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(
          NestedInterface::contract(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<bool on_the_right, typename A, typename B> requires
      interface::contract_in_place_defined_for<Nested, on_the_right, A&&, B&&> or
      interface::contract_in_place_defined_for<Nested, on_the_right, decltype(OpenKalman::nested_object(std::declval<A&&>)), B&&>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<
      interface::contract_in_place_defined_for<Nested, on_the_right, A&&, B&&> or
      interface::contract_in_place_defined_for<Nested, on_the_right, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static decltype(auto)
    contract_in_place(A&& a, B&& b)
    {
      if constexpr (interface::contract_in_place_defined_for<Nested, on_the_right, A&&, B&&>)
      {
        return NestedInterface::template contract_in_place<on_the_right>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        auto&& ret = NestedInterface::template contract_in_place<on_the_right>(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b));
        using Ret = decltype(ret);
        if constexpr (std::is_lvalue_reference_v<Ret> and std::is_same_v<Ret, decltype(OpenKalman::nested_object(std::declval<A&&>()))>)
        {
          return std::forward<A>(a);
        }
        else if constexpr (on_the_right)
        {
          return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(std::forward<Ret>(ret));
        }
        else
        {
          return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<B, 0>, vector_space_descriptor_of_t<A, 1>>(std::forward<Ret>(ret));
        }
      }
    }


#ifdef __cpp_concepts
    template<TriangleType triangle_type, indexible Arg> requires
      interface::cholesky_factor_defined_for<Nested, triangle_type, Arg&&> or
      interface::cholesky_factor_defined_for<Nested, triangle_type, decltype(OpenKalman::nested_object(std::declval<Arg&&>()))>
    static constexpr triangular_matrix<triangle_type> auto
#else
    template<TriangleType triangle_type, typename Arg, std::enable_if_t<
      interface::cholesky_factor_defined_for<Nested, triangle_type, Arg&&> or
      interface::cholesky_factor_defined_for<Nested, triangle_type, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    cholesky_factor(Arg&& arg)
    {
      if constexpr (interface::cholesky_factor_defined_for<Nested, triangle_type, Arg&&>)
      {
        return NestedInterface::template cholesky_factor<triangle_type>(std::forward<Arg>(arg));
      }
      else
      {
        auto tri = NestedInterface::template cholesky_factor<triangle_type>(OpenKalman::nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType significant_triangle, indexible A, indexible U> requires
      interface::rank_update_self_adjoint_defined_for<Nested, significant_triangle, A&&, U&&, const scalar_type_of_t<A>&> or
      interface::rank_update_self_adjoint_defined_for<Nested, significant_triangle, decltype(OpenKalman::nested_object(std::declval<A&&>())), U&&, const scalar_type_of_t<A>&>
    static constexpr hermitian_matrix auto
#else
    template<HermitianAdapterType significant_triangle, typename A, typename U, std::enable_if_t<
      interface::rank_update_self_adjoint_defined_for<Nested, significant_triangle, A&&, U&&, const typename scalar_type_of<A>::type&> or
      interface::rank_update_self_adjoint_defined_for<Nested, significant_triangle, typename nested_object_of<A&&>::type, U&&, const typename scalar_type_of<A>::type&>, int> = 0>
    static constexpr auto
#endif
    rank_update_hermitian(A&& a, U&& u, const scalar_type_of_t<A>& alpha)
    {
      if constexpr (interface::rank_update_self_adjoint_defined_for<Nested, significant_triangle, A&&, U&&, const scalar_type_of_t<A>&>)
      {
        return NestedInterface::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
      }
      else
      {
        auto tri = NestedInterface::template rank_update_hermitian<significant_triangle>(OpenKalman::nested_object(std::forward<A>(a), std::forward<U>(u), alpha));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<TriangleType triangle_type, indexible A, indexible U> requires
      interface::rank_update_triangular_defined_for<Nested, triangle_type, A&&, U&&, const scalar_type_of_t<A>&> or
      interface::rank_update_triangular_defined_for<Nested, triangle_type, decltype(OpenKalman::nested_object(std::declval<A&&>())), U&&, const scalar_type_of_t<A>&>
    static constexpr triangular_matrix<triangle_type> auto
#else
    template<TriangleType triangle_type, typename A, typename U, std::enable_if_t<
      interface::rank_update_triangular_defined_for<Nested, triangle_type, A&&, U&&, const typename scalar_type_of<A>::type&> or
      interface::rank_update_triangular_defined_for<Nested, triangle_type, typename nested_object_of<A&&>::type, U&&, const typename scalar_type_of<A>::type&>, int> = 0>
    static constexpr auto
#endif
    rank_update_triangular(A&& a, U&& u, const scalar_type_of_t<A>& alpha)
    {
      if constexpr (interface::rank_update_triangular_defined_for<Nested, triangle_type, A&&, U&&, const scalar_type_of_t<A>&>)
      {
        return NestedInterface::template rank_update_triangular<triangle_type>(std::forward<A>(a), std::forward<U>(u), alpha);
      }
      else
      {
        auto tri = NestedInterface::template rank_update_triangular<triangle_type>(OpenKalman::nested_object(std::forward<A>(a), std::forward<U>(u), alpha));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<bool must_be_unique, bool must_be_exact, typename A, typename B> requires
      interface::solve_defined_for<Nested, must_be_unique, must_be_exact, A&&, B&&> or
      interface::solve_defined_for<Nested, must_be_unique, must_be_exact, decltype(OpenKalman::nested_object(std::declval<A&&>)), B&&>
    static compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>> auto
#else
    template<bool must_be_unique, bool must_be_exact, typename A, typename B, std::enable_if_t<
      interface::solve_defined_for<Nested, must_be_unique, must_be_exact, A&&, B&&> or
      interface::solve_defined_for<Nested, must_be_unique, must_be_exact, typename nested_object_of<A&&>::type, B&&>, int> = 0>
    static auto
#endif
    solve(A&& a, B&& b)
    {
      if constexpr (interface::solve_defined_for<Nested, must_be_unique, must_be_exact, A&&, B&&>)
      {
        return NestedInterface::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>>(
          NestedInterface::template solve<must_be_unique, must_be_exact>(OpenKalman::nested_object(std::forward<A>(a)), std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<typename A> requires
      interface::LQ_decomposition_defined_for<Nested, A&&> or
      interface::LQ_decomposition_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<A&&>))>
#else
    template<typename A, std::enable_if_t<
      interface::LQ_decomposition_defined_for<Nested, A&&> or
      interface::LQ_decomposition_defined_for<Nested, typename nested_object_of<A&&>::type>, int> = 0>
#endif
    static auto
    LQ_decomposition(A&& a)
    {
      if constexpr (interface::LQ_decomposition_defined_for<Nested, A&&>)
      {
        return NestedInterface::LQ_decomposition(std::forward<A>(a));
      }
      else
      {
        auto&& ret = NestedInterface::LQ_decomposition(OpenKalman::nested_object(std::forward<A>(a)));
        using D0 = vector_space_descriptor_of<A, 0>;
        return internal::make_fixed_square_adapter_like<D0>(std::forward<decltype(ret)>(ret));
      }
    }


#ifdef __cpp_concepts
    template<typename A> requires
      interface::QR_decomposition_defined_for<Nested, A&&> or
      interface::QR_decomposition_defined_for<Nested, decltype(OpenKalman::nested_object(std::declval<A&&>))>
#else
    template<typename A, std::enable_if_t<
      interface::QR_decomposition_defined_for<Nested, A&&> or
      interface::QR_decomposition_defined_for<Nested, typename nested_object_of<A&&>::type>, int> = 0>
#endif
    static auto
    QR_decomposition(A&& a)
    {
      if constexpr (interface::QR_decomposition_defined_for<Nested, A&&>)
      {
        return NestedInterface::QR_decomposition(std::forward<A>(a));
      }
      else
      {
        auto&& ret = NestedInterface::QR_decomposition(OpenKalman::nested_object(std::forward<A>(a)));
        using D1 = vector_space_descriptor_of<A, 1>;
        return internal::make_fixed_square_adapter_like<D1>(std::forward<decltype(ret)>(ret));
      }
    }

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INTERFACES_FIXEDSIZEADAPTER_HPP
