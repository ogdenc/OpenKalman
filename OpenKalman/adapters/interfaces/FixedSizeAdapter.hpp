/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
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

  template<typename NestedObject, typename...Ds>
  struct indexible_object_traits<internal::FixedSizeAdapter<NestedObject, Ds...>>
  {
  private:

    using Xpr = internal::FixedSizeAdapter<NestedObject, Ds...>;


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

    using scalar_type = scalar_type_of_t<NestedObject>;


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
        else return OpenKalman::get_vector_space_descriptor(nested_object(arg), n);
      }
      else if constexpr (equivalent_to<Dimensions<1>, Ds...>)
      {
        return Dimensions<1>{};
      }
      else
      {
        return OpenKalman::get_vector_space_descriptor(nested_object(arg), n);
      }
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nested_object();
    }


#ifdef __cpp_concepts
    template<typename Arg> requires constant_matrix<NestedObject> or (... and equivalent_to<Ds, Dimensions<1>>)
#else
    template<typename M = NestedObject, typename Arg, std::enable_if_t<
      constant_matrix<M> or (... and equivalent_to<Ds, Dimensions<1>>), int> = 0>
#endif
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<NestedObject>) return constant_coefficient {nested_object(arg)};
      else return internal::get_singular_component(arg);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires constant_diagonal_matrix<NestedObject>
#else
    template<typename M = NestedObject, typename Arg, std::enable_if_t<constant_diagonal_matrix<M>, int> = 0>
#endif
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {nested_object(arg)};
    }


    // one_dimensional is not necessary


    // is_square is not necessary


    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<NestedObject, t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<NestedObject, Qualification::depends_on_dynamic_shape>;


    static constexpr bool is_writable = writable<NestedObject>;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires raw_data_defined_for<nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<raw_data_defined_for<typename nested_object_of<Arg&&>::type>, int> = 0>
#endif
    static constexpr decltype(auto)
    raw_data(Arg&& arg)
    {
      return internal::raw_data(OpenKalman::nested_object(std::forward<Arg>(arg)));
    }


    static constexpr Layout layout = layout_of_v<NestedObject>;


#ifdef __cpp_concepts
    template<typename Arg> requires (layout == Layout::stride)
#else
    template<Layout l = layout, typename Arg, std::enable_if_t<l == Layout::stride, int> = 0>
#endif
    static auto
    strides(Arg&& arg)
    {
      return OpenKalman::internal::strides(nested_object(std::forward<Arg>(arg)));
    }

  };


  // ------------------- //
  //  library_interface  //
  // ------------------- //

  template<typename Nested, typename...Ds>
  struct library_interface<internal::FixedSizeAdapter<Nested, Ds...>> : library_interface<std::decay_t<Nested>>
  {
  private:

    using NestedObject = std::decay_t<Nested>;
    using NestedInterface = library_interface<NestedObject>;

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, NestedObject>;

  private:

    template<typename Object, typename Indices>
    static constexpr decltype(auto) add_trailing_indices(const Indices& indices)
    {
      if constexpr (not static_range_size<Indices, Object>)
      {
        constexpr auto N = index_count_v<Object>; //< We know N is not dynamic_size because static_range_size is not satisfied.
        std::array<std::size_t, N> ret;
        std::fill(std::copy(indices.begin(), indices.end(), ret.begin()), ret.end(), 0);
        return ret;
      }
      else return indices;
    }

  public:

#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      interface::get_component_defined_for<NestedObject, nested_object_of_t<Arg&>, std::initializer_list<std::size_t>>
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::get_component_defined_for<NestedObject, typename nested_object_of<Arg&>::type, std::initializer_list<std::size_t>>, int> = 0>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      return NestedInterface::get_component(nested_object(std::forward<Arg>(arg)), add_trailing_indices<NestedObject>(indices));
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      interface::set_component_defined_for<NestedObject, nested_object_of_t<Arg&>, const scalar_type_of_t<Arg>&, std::initializer_list<std::size_t>>
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::set_component_defined_for<NestedObject, typename nested_object_of<Arg&>::type, const typename scalar_type_of<Arg>::type&, std::initializer_list<std::size_t>>, int> = 0>
#endif
    static constexpr void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      NestedInterface::set_component(nested_object(arg), s, add_trailing_indices<NestedObject>(indices));
    }


#ifdef __cpp_concepts
    template<typename A> requires interface::to_native_matrix_defined_for<NestedObject, nested_object_of_t<A&&>>
#else
    template<typename A, std::enable_if_t<interface::to_native_matrix_defined_for<NestedObject, nested_object_of_t<A&&>>, int> = 0>
#endif
    static decltype(auto)
    to_native_matrix(A&& a)
    {
      return internal::make_fixed_size_adapter<Ds...>(NestedInterface::to_native_matrix(nested_object(std::forward<A>(a))));
    }


#ifdef __cpp_concepts
    template<typename To, typename From> requires
      interface::assign_defined_for<NestedObject, nested_object_of_t<To&>, From&&>
#else
    template<typename To, typename From, std::enable_if_t<
      interface::assign_defined_for<NestedObject, nested_object_of_t<To&>, From&&>, int> = 0>
#endif
    static void
    assign(To& a, From&& b)
    {
      NestedInterface::assign(nested_object(a), std::forward<From>(b));
    }


#ifdef __cpp_concepts
    template<Layout layout, typename Scalar, typename...D> requires
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&...>
#else
    template<Layout layout, typename Scalar, typename...D, std::enable_if_t<
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&...>, int> = 0>
#endif
    static auto
    make_default(D&&...d)
    {
      return NestedInterface::template make_default<layout, Scalar>(std::forward<D>(d)...);
    }


#ifdef __cpp_concepts
    template<Layout layout, typename Arg, typename...Scalars> requires
      interface::fill_components_defined_for<NestedObject, layout, nested_object_of_t<Arg&>, Scalars...>
#else
    template<Layout layout, typename Arg, typename...Scalars, std::enable_if_t<
      interface::fill_components_defined_for<NestedObject, layout, typename nested_object_of<Arg&>::type, Scalars...>, int> = 0>
#endif
    static void
    fill_components(Arg& arg, const Scalars...scalars)
    {
      NestedInterface::template fill_components<layout>(nested_object(arg), scalars...);
    }


#ifdef __cpp_concepts
    template<typename C, typename...D> requires interface::make_constant_matrix_defined_for<NestedObject, C&&, D&&...>
#else
    template<typename C, typename...D, std::enable_if_t<interface::make_constant_matrix_defined_for<NestedObject, C&&, D&&...>, int> = 0>
#endif
    static constexpr auto
    make_constant(C&& c, D&&...d)
    {
      return NestedInterface::make_constant(std::forward<C>(c), std::forward<D>(d)...);
    }


#ifdef __cpp_concepts
    template<typename Scalar, typename...D> requires interface::make_identity_matrix_defined_for<NestedObject, Scalar, D&&...>
#else
    template<typename Scalar, typename...D, std::enable_if_t<interface::make_identity_matrix_defined_for<NestedObject, Scalar, D&&...>, int> = 0>
#endif
    static constexpr auto
    make_identity_matrix(D&&...d)
    {
      return NestedInterface::make_identity_matrix(std::forward<D>(d)...);
    }


#ifdef __cpp_concepts
    template<TriangleType t, indexible Arg> requires
      interface::make_triangular_matrix_defined_for<NestedObject, t, nested_object_of_t<Arg&&>>
    static constexpr triangular_matrix<t> auto
#else
    template<TriangleType t, typename Arg, std::enable_if_t<
      interface::make_triangular_matrix_defined_for<NestedObject, t, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    make_triangular_matrix(Arg&& arg)
    {
      return internal::make_fixed_size_adapter<Ds...>(NestedInterface::template make_triangular_matrix<t>(nested_object(std::forward<Arg>(arg))));
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType t, indexible Arg> requires
      interface::make_hermitian_adapter_defined_for<NestedObject, t, nested_object_of_t<Arg&&>>
    static constexpr hermitian_matrix auto
#else
    template<HermitianAdapterType t, typename Arg, std::enable_if_t<
      make_hermitian_adapter_defined_for<NestedObject, t, typename nested_object_of<Arg&>::type>, int> = 0>
    static constexpr auto
#endif
    make_hermitian_adapter(Arg&& arg)
    {
      return internal::make_fixed_size_adapter<Ds...>(NestedInterface::template make_hermitian_adapter<t>(nested_object(std::forward<Arg>(arg))));
    }


#ifdef __cpp_concepts
    template<typename Arg, typename...Begin, typename...Size> requires
      interface::get_slice_defined_for<NestedObject, nested_object_of_t<Arg&&>, const std::tuple<Begin...>&, const std::tuple<Size...>&>
#else
    template<typename Arg, typename...Begin, typename...Size, std::enable_if_t<
      interface::get_slice_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, const std::tuple<Begin...>&, const std::tuple<Size...>&>, int> = 0>
#endif
    static decltype(auto)
    get_slice(Arg&& arg, const std::tuple<Begin...>& begin_tup, const std::tuple<Size...>& size_tup)
    {
      return internal::make_fixed_size_adapter<decltype(get_vector_space_descriptor_slice(std::declval<Begin>(), std::declval<Size>()))...>(
        NestedInterface::get_slice(nested_object(std::forward<Arg>(arg)), begin_tup, size_tup));
    };


#ifdef __cpp_concepts
    template<typename Arg, typename Block, typename...Begin> requires
      interface::set_slice_defined_for<Arg, nested_object_of_t<Arg&>, Block&&, const Begin&...>
#else
    template<typename Arg, typename Block, typename...Begin, std::enable_if_t<
      interface::set_slice_defined_for<Arg, typename nested_object_of<Arg&>::type, Block&&, const Begin&...>, int> = 0>
#endif
    static void
    set_slice(Arg& arg, Block&& block, const Begin&...begin)
    {
      NestedInterface::set_slice(nested_object(arg), std::forward<Block>(block), begin...);
    };


#ifdef __cpp_concepts
    template<TriangleType t, typename A, typename B> requires
      interface::set_triangle_defined_for<NestedObject, t, nested_object_of_t<A&&>, B&&>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      interface::set_triangle_defined_for<NestedObject, t, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static void
    set_triangle(A&& a, B&& b)
    {
      NestedInterface::template set_triangle<t>(nested_object(std::forward<A>(a)), std::forward<B>(b));
    }


#ifdef __cpp_concepts
    template<typename Arg> requires
      interface::to_diagonal_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr diagonal_matrix auto
#else
    template<typename Arg, std::enable_if_t<
      interface::to_diagonal_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    to_diagonal(Arg&& arg)
    {
      using D = vector_space_descriptor_of_t<Arg, 0>;
      return internal::make_fixed_square_adapter_like<D>(NestedInterface::to_diagonal(nested_object(std::forward<Arg>(arg))));
    }

  private:

    template<typename Arg, typename V0, typename V1, typename...Vs>
    static constexpr decltype(auto)
    diagonal_of_impl(Arg&& arg, const std::tuple<V0, V1, Vs...>&)
    {
      using D0 = decltype(internal::smallest_vector_space_descriptor<scalar_type_of_t<Arg>>(std::declval<V0>(), std::declval<V1>()));
      return make_fixed_size_adapter<D0, Vs...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::diagonal_of_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr vector auto
#else
    template<typename Arg, std::enable_if_t<
      interface::diagonal_of_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    diagonal_of(Arg&& arg)
    {
      return diagonal_of_impl(std::forward<Arg>(arg),
        std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{Dimensions<1>{}, Dimensions<1>{}}));
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
      interface::broadcast_defined_for<NestedObject, nested_object_of_t<Arg&&>, const Factors&...>
    static indexible auto
#else
    template<typename Arg, typename...Factors, std::enable_if_t<
      interface::broadcast_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, const Factors&...>, int> = 0>
    static auto
#endif
    broadcast(Arg&& arg, const Factors&...factors)
    {
      auto&& ret = NestedInterface::broadcast(nested_object(std::forward<Arg>(arg)), factors...);
      using Ret = decltype(ret);
      auto seq = std::make_index_sequence<std::max(index_count_v<Arg>, sizeof...(factors))>{};
      return broadcast_impl(std::forward<Ret>(ret), seq, std::forward_as_tuple(factors...));
    }


#ifdef __cpp_concepts
    template<vector_space_descriptor...IDs, typename Operation> requires
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<IDs...>&, Operation&&>
    static indexible auto
#else
    template<typename...IDs, typename Operation, std::enable_if_t<
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<IDs...>&, Operation&&>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<IDs...>& d_tup, Operation&& op)
    {
      return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op));
    }


#ifdef __cpp_concepts
    template<vector_space_descriptor...IDs, typename Operation, indexible Arg, indexible...Args> requires
      interface::n_ary_operation_defined_for<NestedObject, const std::tuple<IDs...>&, Operation&&, nested_object_of_t<Arg&&>, Args...>
    static indexible auto
#else
    template<typename...IDs, typename Operation, typename Arg, typename...Args, std::enable_if_t<
      interface::n_ary_operation_defined_for<NestedObject, const std::tuple<IDs...>&, Operation&&, typename nested_object_of<Arg&&>::type, Args...>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<IDs...>& d_tup, Operation&& op, Arg&& arg, Args&&...args)
    {
      return internal::make_fixed_size_adapter<IDs...>(
        NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op), nested_object(std::forward<Arg>(arg)), std::forward<Args>(args)...));
    }

  private:

    template<std::size_t...indices, typename Arg, std::size_t...Ix>
    static constexpr decltype(auto)
    reduce_impl(Arg&& arg, std::index_sequence<Ix...> seq)
    {
      return internal::make_fixed_size_adapter<
        std::conditional_t<
          []{ constexpr auto I = Ix; return ((I == indices) or ...); },
          uniform_fixed_vector_space_descriptor_component_of_t<vector_space_descriptor_of_t<Arg, Ix>>,
          Ds>...>
        (std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, indexible Arg> requires
      interface::reduce_defined_for<NestedObject, BinaryFunction&&, nested_object_of_t<Arg&&>, indices...>
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
      interface::reduce_defined_for<NestedObject, BinaryFunction&&, typename nested_object_of<Arg&&>::type, indices...>, int> = 0>
#endif
    static constexpr decltype(auto)
    reduce(BinaryFunction&& op, Arg&& arg)
    {
      return reduce_impl<indices...>(
        NestedInterface::template reduce<indices...>(std::forward<BinaryFunction>(op), nested_object(std::forward<Arg>(arg))),
        std::index_sequence_for<Ds...>{});
    }

  private:

    template<typename V0, typename...Vs, typename Arg>
    static constexpr decltype(auto)
    to_euclidean_impl(Arg&& arg)
    {
      using V = Dimensions<euclidean_dimension_size_of_v<V0>>;
      return internal::make_fixed_size_adapter<V, Vs...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg> requires interface::to_euclidean_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::to_euclidean_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    to_euclidean(Arg&& arg)
    {
      return to_euclidean_impl<Ds...>(NestedInterface::to_euclidean(nested_object(std::forward<Arg>(arg))));
    }

  private:

    template<typename V0, typename V, typename...Vs, typename Arg>
    static constexpr decltype(auto)
    from_euclidean_impl(Arg&& arg)
    {
      return internal::make_fixed_size_adapter<V0, Vs...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor V> requires
      interface::from_euclidean_defined_for<NestedObject, nested_object_of_t<Arg&&>, V&&>
    static constexpr indexible auto
#else
    template<typename Arg, typename V, std::enable_if_t<
      interface::from_euclidean_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, V&&>, int> = 0>
    static constexpr auto
#endif
    from_euclidean(Arg&& arg, V&& v)
    {
      return from_euclidean_impl<V, Ds...>(NestedInterface::from_euclidean(nested_object(std::forward<Arg>(arg)), std::forward<V>(v)));
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires interface::wrap_angles_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::wrap_angles_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    wrap_angles(Arg&& arg)
    {
      return internal::make_fixed_size_adapter<Ds...>(NestedInterface::wrap_angles(nested_object(std::forward<Arg>(arg))));
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::conjugate_defined_for<NestedObject, Arg&&> or
      interface::conjugate_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::conjugate_defined_for<NestedObject, Arg&&> or
      interface::conjugate_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    conjugate(Arg&& arg)
    {
      if constexpr (interface::conjugate_defined_for<NestedObject, Arg&&>)
      {
        return NestedInterface::conjugate(std::forward<Arg>(arg));
      }
      else
      {
        auto&& conj = NestedInterface::conjugate(nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_size_adapter_like<Arg>(std::forward<decltype(conj)>(conj));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::transpose_defined_for<NestedObject, Arg&&> or
      interface::transpose_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::transpose_defined_for<NestedObject, Arg&&> or
      interface::transpose_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    transpose(Arg&& arg)
    {
      if constexpr (interface::transpose_defined_for<NestedObject, Arg&&>)
      {
        return NestedInterface::transpose(std::forward<Arg>(arg));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>>(
          NestedInterface::transpose(nested_object(std::forward<Arg>(arg))));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::adjoint_defined_for<NestedObject, Arg&&> or
      interface::adjoint_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::adjoint_defined_for<NestedObject, Arg&&> or
      interface::adjoint_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    adjoint(Arg&& arg)
    {
      if constexpr (interface::adjoint_defined_for<NestedObject, Arg&&>)
      {
        return NestedInterface::adjoint(std::forward<Arg>(arg));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>>(
          NestedInterface::adjoint(nested_object(std::forward<Arg>(arg))));
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::determinant_defined_for<NestedObject, Arg&&> or
      interface::determinant_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
    template<typename Arg, std::enable_if_t<
      interface::determinant_defined_for<NestedObject, Arg&&> or
      interface::determinant_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    determinant(Arg&& arg)
    {
      if constexpr (interface::determinant_defined_for<NestedObject, Arg&&>)
      {
        return NestedInterface::determinant(std::forward<Arg>(arg));
      }
      else
      {
        return NestedInterface::determinant(nested_object(std::forward<Arg>(arg)));
      }
    }


#ifdef __cpp_concepts
    template<typename Arg, typename...Args> requires
      interface::sum_defined_for<NestedObject, Arg&&, Args&&...> or
      interface::sum_defined_for<NestedObject, nested_object_of_t<Arg&&>, Args&&...>
#else
    template<typename Arg, typename...Args, std::enable_if_t<
      interface::sum_defined_for<NestedObject, Arg&&, Args&&...> or
      interface::sum_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, Args&&...>, int> = 0>
#endif
    static auto
    sum(Arg&& arg, Args&&...args)
    {
      if constexpr (interface::sum_defined_for<NestedObject, Arg&&, Args&&...>)
      {
        return NestedInterface::sum(std::forward<Arg>(arg), std::forward<Args>(args)...);
      }
      else
      {
        return NestedInterface::sum(nested_object(std::forward<Arg>(arg)), std::forward<Args>(args)...);
      }
    }


#ifdef __cpp_concepts
    template<typename A, typename B> requires
      interface::contract_defined_for<NestedObject, A&&, B&&> or
      interface::contract_defined_for<NestedObject, nested_object_of_t<A&&>, B&&>
#else
    template<typename A, typename B, std::enable_if_t<
      interface::contract_defined_for<NestedObject, A&&, B&&> or
      interface::contract_defined_for<NestedObject, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static auto
    contract(A&& a, B&& b)
    {
      if constexpr (interface::contract_defined_for<NestedObject, A&&, B&&>)
      {
        return NestedInterface::contract(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 0>, vector_space_descriptor_of_t<B, 1>>(
          NestedInterface::contract(nested_object(std::forward<A>(a)), std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<bool on_the_right, typename A, typename B> requires
      interface::contract_in_place_defined_for<NestedObject, on_the_right, A&&, B&&> or
      interface::contract_in_place_defined_for<NestedObject, on_the_right, nested_object_of_t<A&&>, B&&>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<
      interface::contract_in_place_defined_for<NestedObject, on_the_right, A&&, B&&> or
      interface::contract_in_place_defined_for<NestedObject, on_the_right, typename nested_object_of<A&&>::type, B&&>, int> = 0>
#endif
    static decltype(auto)
    contract_in_place(A&& a, B&& b)
    {
      if constexpr (interface::contract_in_place_defined_for<NestedObject, on_the_right, A&&, B&&>)
      {
        return NestedInterface::template contract_in_place<on_the_right>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        auto&& ret = NestedInterface::template contract_in_place<on_the_right>(nested_object(std::forward<A>(a)), std::forward<B>(b));
        using Ret = decltype(ret);
        if constexpr (std::is_lvalue_reference_v<Ret> and std::is_same_v<Ret, nested_object_of_t<A&&>>)
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
      interface::cholesky_factor_defined_for<NestedObject, triangle_type, Arg&&> or
      interface::cholesky_factor_defined_for<NestedObject, triangle_type, nested_object_of_t<Arg&&>>
    static constexpr triangular_matrix<triangle_type> auto
#else
    template<TriangleType triangle_type, typename Arg, std::enable_if_t<
      interface::cholesky_factor_defined_for<NestedObject, triangle_type, Arg&&> or
      interface::cholesky_factor_defined_for<NestedObject, triangle_type, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    cholesky_factor(Arg&& arg)
    {
      if constexpr (interface::cholesky_factor_defined_for<NestedObject, triangle_type, Arg&&>)
      {
        return NestedInterface::template cholesky_factor<triangle_type>(std::forward<Arg>(arg));
      }
      else
      {
        auto tri = NestedInterface::template cholesky_factor<triangle_type>(nested_object(std::forward<Arg>(arg)));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<HermitianAdapterType significant_triangle, indexible A, indexible U> requires
      interface::rank_update_self_adjoint_defined_for<NestedObject, significant_triangle, A&&, U&&, const scalar_type_of_t<A>&> or
      interface::rank_update_self_adjoint_defined_for<NestedObject, significant_triangle, nested_object_of_t<A&&>, U&&, const scalar_type_of_t<A>&>
    static constexpr hermitian_matrix auto
#else
    template<HermitianAdapterType significant_triangle, typename A, typename U, std::enable_if_t<
      interface::rank_update_self_adjoint_defined_for<NestedObject, significant_triangle, A&&, U&&, const typename scalar_type_of<A>::type&> or
      interface::rank_update_self_adjoint_defined_for<NestedObject, significant_triangle, typename nested_object_of<A&&>::type, U&&, const typename scalar_type_of<A>::type&>, int> = 0>
    static constexpr auto
#endif
    rank_update_hermitian(A&& a, U&& u, const scalar_type_of_t<A>& alpha)
    {
      if constexpr (interface::rank_update_self_adjoint_defined_for<NestedObject, significant_triangle, A&&, U&&, const scalar_type_of_t<A>&>)
      {
        return NestedInterface::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
      }
      else
      {
        auto tri = NestedInterface::template rank_update_hermitian<significant_triangle>(nested_object(std::forward<A>(a), std::forward<U>(u), alpha));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<TriangleType triangle_type, indexible A, indexible U> requires
      interface::rank_update_triangular_defined_for<NestedObject, triangle_type, A&&, U&&, const scalar_type_of_t<A>&> or
      interface::rank_update_triangular_defined_for<NestedObject, triangle_type, nested_object_of_t<A&&>, U&&, const scalar_type_of_t<A>&>
    static constexpr triangular_matrix<triangle_type> auto
#else
    template<TriangleType triangle_type, typename A, typename U, std::enable_if_t<
      interface::rank_update_triangular_defined_for<NestedObject, triangle_type, A&&, U&&, const typename scalar_type_of<A>::type&> or
      interface::rank_update_triangular_defined_for<NestedObject, triangle_type, typename nested_object_of<A&&>::type, U&&, const typename scalar_type_of<A>::type&>, int> = 0>
    static constexpr auto
#endif
    rank_update_triangular(A&& a, U&& u, const scalar_type_of_t<A>& alpha)
    {
      if constexpr (interface::rank_update_triangular_defined_for<NestedObject, triangle_type, A&&, U&&, const scalar_type_of_t<A>&>)
      {
        return NestedInterface::template rank_update_triangular<triangle_type>(std::forward<A>(a), std::forward<U>(u), alpha);
      }
      else
      {
        auto tri = NestedInterface::template rank_update_triangular<triangle_type>(nested_object(std::forward<A>(a), std::forward<U>(u), alpha));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<bool must_be_unique, bool must_be_exact, typename A, typename B> requires
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, A&&, B&&> or
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, nested_object_of_t<A&&>, B&&>
    static compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>> auto
#else
    template<bool must_be_unique, bool must_be_exact, typename A, typename B, std::enable_if_t<
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, A&&, B&&> or
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, typename nested_object_of<A&&>::type, B&&>, int> = 0>
    static auto
#endif
    solve(A&& a, B&& b)
    {
      if constexpr (interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, A&&, B&&>)
      {
        return NestedInterface::template solve<must_be_unique, must_be_exact>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        return internal::make_fixed_size_adapter<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>>(
          NestedInterface::template solve<must_be_unique, must_be_exact>(nested_object(std::forward<A>(a)), std::forward<B>(b)));
      }
    }


#ifdef __cpp_concepts
    template<typename A> requires
      interface::LQ_decomposition_defined_for<NestedObject, A&&> or
      interface::LQ_decomposition_defined_for<NestedObject, nested_object_of_t<A&&>>
#else
    template<typename A, std::enable_if_t<
      interface::LQ_decomposition_defined_for<NestedObject, A&&> or
      interface::LQ_decomposition_defined_for<NestedObject, typename nested_object_of<A&&>::type>, int> = 0>
#endif
    static auto
    LQ_decomposition(A&& a)
    {
      if constexpr (interface::LQ_decomposition_defined_for<NestedObject, A&&>)
      {
        return NestedInterface::LQ_decomposition(std::forward<A>(a));
      }
      else
      {
        auto&& ret = NestedInterface::LQ_decomposition(nested_object(std::forward<A>(a)));
        using D0 = vector_space_descriptor_of<A, 0>;
        return internal::make_fixed_square_adapter_like<D0>(std::forward<decltype(ret)>(ret));
      }
    }


#ifdef __cpp_concepts
    template<typename A> requires
      interface::QR_decomposition_defined_for<NestedObject, A&&> or
      interface::QR_decomposition_defined_for<NestedObject, nested_object_of_t<A&&>>
#else
    template<typename A, std::enable_if_t<
      interface::QR_decomposition_defined_for<NestedObject, A&&> or
      interface::QR_decomposition_defined_for<NestedObject, typename nested_object_of<A&&>::type>, int> = 0>
#endif
    static auto
    QR_decomposition(A&& a)
    {
      if constexpr (interface::QR_decomposition_defined_for<NestedObject, A&&>)
      {
        return NestedInterface::QR_decomposition(std::forward<A>(a));
      }
      else
      {
        auto&& ret = NestedInterface::QR_decomposition(nested_object(std::forward<A>(a)));
        using D1 = vector_space_descriptor_of<A, 1>;
        return internal::make_fixed_square_adapter_like<D1>(std::forward<decltype(ret)>(ret));
      }
    }

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INTERFACES_FIXEDSIZEADAPTER_HPP
