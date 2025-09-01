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

  template<typename NestedObject, typename Descriptors>
  struct indexible_object_traits<internal::FixedSizeAdapter<NestedObject, Descriptors>>
  {
  private:

    using Xpr = internal::FixedSizeAdapter<NestedObject, Descriptors>;

  public:

    using scalar_type = scalar_type_of_t<NestedObject>;


    template<typename Arg>
    static constexpr auto count_indices(const Arg&)
    {
      // Truncate any trailing ℝ¹ dimensions
      using NewDesc = decltype(OpenKalman::coordinates::internal::strip_1D_tail(std::declval<Descriptors>()));
      return collections::size_of<NewDesc>{};
    }


    template<typename Arg, typename N>
    static constexpr auto get_pattern_collection(Arg&& arg, const N& n)
    {
      constexpr auto dim = decltype(count_indices(arg))::value;
      if constexpr (values::fixed<N>)
      {
        if constexpr (N::value >= dim)
          return coordinates::Axis{};
        else if constexpr (fixed_pattern<collections::collection_element_t<N::value, Descriptors>>)
          return collections::collection_element_t<N::value, Descriptors> {};
        else
          return OpenKalman::get_pattern_collection(std::forward<Arg>(arg).nested_object(), n);
      }
      else
      {
        return OpenKalman::get_pattern_collection(std::forward<Arg>(arg).nested_object(), n);
      }
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nested_object();
    }


#ifdef __cpp_concepts
    template<typename Arg> requires constant_matrix<NestedObject>
#else
    template<typename M = NestedObject, typename Arg, std::enable_if_t<constant_matrix<M>, int> = 0>
#endif
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {nested_object(arg)};
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


    template<triangle_type t>
    static constexpr bool is_triangular = triangular_matrix<NestedObject, t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<NestedObject, applicability::permitted>;


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


    static constexpr data_layout layout = layout_of_v<NestedObject>;


#ifdef __cpp_concepts
    template<typename Arg> requires (layout == data_layout::stride)
#else
    template<data_layout l = layout, typename Arg, std::enable_if_t<l == data_layout::stride, int> = 0>
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

  template<typename Nested, typename Descriptors>
  struct library_interface<internal::FixedSizeAdapter<Nested, Descriptors>> : library_interface<std::decay_t<Nested>>
  {
  private:

    using NestedObject = std::decay_t<Nested>;
    using NestedInterface = library_interface<NestedObject>;

  public:

    template<typename Derived>
    using library_base = internal::library_base_t<Derived, NestedObject>;

  private:

    template<typename Object, typename Indices>
    static constexpr decltype(auto) add_trailing_indices(const Indices& indices)
    {
      if constexpr (not index_collection_for<Indices, Object>)
      {
        constexpr auto N = index_count_v<Object>; //< We know N is not dynamic_size because index_collection_for is not satisfied.
        std::array<std::size_t, N> ret;
        std::ranges::fill(std::ranges::copy<stdcompat::ranges::begin(indices), stdcompat::ranges::end(indices), stdcompat::ranges::begin(ret)), stdcompat::ranges::end(ret), 0);
        return ret;
      }
      else return indices;
    }

  public:

#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>> and
      interface::get_component_defined_for<NestedObject, nested_object_of_t<Arg&>, std::initializer_list<std::size_t>>
    static constexpr values::scalar decltype(auto)
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
    template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>> and
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
      return internal::make_fixed_size_adapter<Descriptors>(NestedInterface::to_native_matrix(nested_object(std::forward<A>(a))));
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
    template<data_layout layout, typename Scalar, typename D> requires
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&>
#else
    template<data_layout layout, typename Scalar, typename D, std::enable_if_t<
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&>, int> = 0>
#endif
    static auto
    make_default(D&& d)
    {
      return NestedInterface::template make_default<layout, Scalar>(std::forward<D>(d));
    }


#ifdef __cpp_concepts
    template<data_layout layout, typename Arg, typename...Scalars> requires
      interface::fill_components_defined_for<NestedObject, layout, nested_object_of_t<Arg&>, Scalars...>
#else
    template<data_layout layout, typename Arg, typename...Scalars, std::enable_if_t<
      interface::fill_components_defined_for<NestedObject, layout, typename nested_object_of<Arg&>::type, Scalars...>, int> = 0>
#endif
    static void
    fill_components(Arg& arg, const Scalars...scalars)
    {
      NestedInterface::template fill_components<layout>(nested_object(arg), scalars...);
    }


#ifdef __cpp_concepts
    template<typename C, typename D> requires interface::make_constant_defined_for<NestedObject, C&&, D&&>
#else
    template<typename C, typename D, std::enable_if_t<interface::make_constant_defined_for<NestedObject, C&&, D&&>, int> = 0>
#endif
    static constexpr auto
    make_constant(C&& c, D&& d)
    {
      return NestedInterface::make_constant(std::forward<C>(c), std::forward<D>(d));
    }


#ifdef __cpp_concepts
    template<typename Scalar, typename D> requires interface::make_identity_matrix_defined_for<NestedObject, Scalar, D&&>
#else
    template<typename Scalar, typename D, std::enable_if_t<interface::make_identity_matrix_defined_for<NestedObject, Scalar, D&&>, int> = 0>
#endif
    static constexpr auto
    make_identity_matrix(D&& d)
    {
      return NestedInterface::make_identity_matrix(std::forward<D>(d));
    }


#ifdef __cpp_concepts
    template<triangle_type t, indexible Arg> requires
      interface::make_triangular_matrix_defined_for<NestedObject, t, nested_object_of_t<Arg&&>>
    static constexpr triangular_matrix<t> auto
#else
    template<triangle_type t, typename Arg, std::enable_if_t<
      interface::make_triangular_matrix_defined_for<NestedObject, t, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    make_triangular_matrix(Arg&& arg)
    {
      return internal::make_fixed_size_adapter<Descriptors>(NestedInterface::template make_triangular_matrix<t>(nested_object(std::forward<Arg>(arg))));
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
      return internal::make_fixed_size_adapter<Descriptors>(NestedInterface::template make_hermitian_adapter<t>(nested_object(std::forward<Arg>(arg))));
    }

  private:

    template<typename Arg, typename...Begin, typename...Size, std::size_t...Ix>
    static decltype(auto)
    get_slice_impl(Arg&& arg, const std::tuple<Begin...>& begin_tup, const std::tuple<Size...>& size_tup, std::index_sequence<Ix...>)
    {
      using NewDesc = std::tuple<std::decay_t<decltype(coordinates::get_slice<scalar_type_of_t<Arg>>(
        std::declval<collections::collection_element_t<Ix, Descriptors>>, std::declval<Begin>(), std::declval<Size>()))>...>;
      return internal::make_fixed_size_adapter<NewDesc>(NestedInterface::get_slice(nested_object(std::forward<Arg>(arg)), begin_tup, size_tup));
    }

  public:

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
      return get_slice_impl(std::forward<Arg>(arg), begin_tup, size_tup, std::make_index_sequence<collections::size_of_v<Descriptors>>{});
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
    template<triangle_type t, typename A, typename B> requires
      interface::set_triangle_defined_for<NestedObject, t, nested_object_of_t<A&&>, B&&>
#else
    template<triangle_type t, typename A, typename B, std::enable_if_t<
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
      using D0 = decltype(internal::smallest_pattern<scalar_type_of_t<Arg>>(std::declval<V0>(), std::declval<V1>()));
      return OpenKalman::internal::make_fixed_size_adapter<D0, Vs...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg> requires (diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject>) or
      (diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject, 1> and
        interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>) or
      interface::diagonal_of_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<(diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject>) or
      (diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject, 1> and
        interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>) or 
      interface::diagonal_of_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    diagonal_of(Arg&& arg)
    {
      if constexpr (diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject>)
        return diagonal_of_impl(nested_object(nested_object(std::forward<Arg>(arg))));
      else if constexpr (diagonal_matrix<NestedObject> and internal::has_nested_vector<NestedObject, 1> and
          interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>)
        return diagonal_of_impl(NestedInterface::transpose(nested_object(nested_object(std::forward<Arg>(arg)))));
      else 
        return diagonal_of_impl(NestedInterface::diagonal_of(nested_object(std::forward<Arg>(arg))),
          std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{coordinates::Axis{}, coordinates::Axis{}}));
    }

  private:

    template<std::size_t Ix, typename Arg, typename Factors_tup>
    static constexpr auto broadcast_for_index(const Arg& arg, const Factors_tup& factors_tup)
    {
      constexpr auto N = collections::size_of_v<Factors_tup>;
      if constexpr (Ix < N)
        return get_pattern_collection<Ix>(arg) * std::get<Ix>(factors_tup);
      else
        return coordinates::Axis{};
    }


    template<typename Arg, std::size_t...Is, typename Factors_tup>
    static constexpr auto broadcast_impl(Arg&& arg, std::index_sequence<Is...>, const Factors_tup& factors_tup)
    {
      constexpr auto N = collections::size_of_v<Factors_tup>;
      return internal::make_fixed_size_adapter<decltype(broadcast_for_index<Is>(arg, factors_tup))...>(std::forward<Arg>(arg));
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg, values::index...Factors> requires
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
    template<coordinates::pattern...IDs, typename Operation> requires
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
    template<coordinates::pattern...IDs, typename Operation, indexible Arg, indexible...Args> requires
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

    template<std::size_t Ix, std::size_t...indices>
    static constexpr bool matching_Ix() { return ((Ix == indices) or ...); }

    template<std::size_t...indices, typename Arg, std::size_t...Ix>
    static constexpr decltype(auto)
    reduce_impl(Arg&& arg, std::index_sequence<Ix...> seq)
    {
      return internal::make_fixed_size_adapter<std::tuple<
        std::conditional_t<
          matching_Ix<Ix, indices...>(),
          uniform_pattern_component_of_t<vector_space_descriptor_of_t<Arg, Ix>>,
          collections::collection_element_t<Ix, Descriptors>>...>>
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
        std::make_index_sequence<collections::size_of_v<Descriptors>>{});
    }


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
      constexpr auto dim = collections::size_of_v<Descriptors>;
      if constexpr (dim == 0)
      {
        return std::forward<Arg>(arg);
      }
      else
      {
        using D0 = collections::collection_element_t<0, Descriptors>;
        if constexpr (coordinates::euclidean_pattern<D0>)
        {
          return std::forward<Arg>(arg);
        }
        else
        {
          using V0 = std::conditional_t<
            fixed_pattern<D0>,
            coordinates::Dimensions<coordinates::stat_dimension_of_v<D0>>,
            coordinates::DynamicDescriptor<scalar_type_of_t<Arg>>>;
          using Vtail = std::decay_t<decltype(internal::tuple_slice<1, dim>(std::declval<Descriptors>()))>;
          using Vcat = decltype(std::tuple_cat(std::declval<V0>()), std::declval<Vtail>());
          return internal::make_fixed_size_adapter<Vcat>(NestedInterface::to_euclidean(nested_object(std::forward<Arg>(arg))));
        }
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg, coordinates::pattern D> requires
      interface::from_euclidean_defined_for<NestedObject, nested_object_of_t<Arg&&>, D&&>
    static constexpr indexible auto
#else
    template<typename Arg, typename V, std::enable_if_t<
      interface::from_euclidean_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, V&&>, int> = 0>
    static constexpr auto
#endif
    from_euclidean(Arg&& arg, D&& d)
    {
      if constexpr (coordinates::euclidean_pattern<D>)
      {
        return std::forward<Arg>(arg);
      }
      else
      {
        constexpr auto dim = collections::size_of_v<Descriptors>;
        using Vtail = std::decay_t<decltype(internal::tuple_slice<1, dim>(std::declval<Descriptors>()))>;
        using Vcat = decltype(std::tuple_cat(std::declval<D>()), std::declval<Vtail>());
        return internal::make_fixed_size_adapter<Vcat>(NestedInterface::from_euclidean(nested_object(std::forward<Arg>(arg)), std::forward<D>(d)));
      }
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
      return internal::make_fixed_size_adapter<Descriptors>(NestedInterface::wrap_angles(nested_object(std::forward<Arg>(arg))));
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
    template<triangle_type tri, indexible Arg> requires
      interface::cholesky_factor_defined_for<NestedObject, tri, Arg&&> or
      interface::cholesky_factor_defined_for<NestedObject, tri, nested_object_of_t<Arg&&>>
    static constexpr triangular_matrix<tri> auto
#else
    template<triangle_type tri, typename Arg, std::enable_if_t<
      interface::cholesky_factor_defined_for<NestedObject, tri, Arg&&> or
      interface::cholesky_factor_defined_for<NestedObject, tri, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    cholesky_factor(Arg&& arg)
    {
      if constexpr (interface::cholesky_factor_defined_for<NestedObject, tri, Arg&&>)
      {
        return NestedInterface::template cholesky_factor<tri>(std::forward<Arg>(arg));
      }
      else
      {
        auto tri = NestedInterface::template cholesky_factor<tri>(nested_object(std::forward<Arg>(arg)));
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
    template<triangle_type tri, indexible A, indexible U> requires
      interface::rank_update_triangular_defined_for<NestedObject, tri, A&&, U&&, const scalar_type_of_t<A>&> or
      interface::rank_update_triangular_defined_for<NestedObject, tri, nested_object_of_t<A&&>, U&&, const scalar_type_of_t<A>&>
    static constexpr triangular_matrix<tri> auto
#else
    template<triangle_type tri, typename A, typename U, std::enable_if_t<
      interface::rank_update_triangular_defined_for<NestedObject, tri, A&&, U&&, const typename scalar_type_of<A>::type&> or
      interface::rank_update_triangular_defined_for<NestedObject, tri, typename nested_object_of<A&&>::type, U&&, const typename scalar_type_of<A>::type&>, int> = 0>
    static constexpr auto
#endif
    rank_update_triangular(A&& a, U&& u, const scalar_type_of_t<A>& alpha)
    {
      if constexpr (interface::rank_update_triangular_defined_for<NestedObject, tri, A&&, U&&, const scalar_type_of_t<A>&>)
      {
        return NestedInterface::template rank_update_triangular<tri>(std::forward<A>(a), std::forward<U>(u), alpha);
      }
      else
      {
        auto tri = NestedInterface::template rank_update_triangular<tri>(nested_object(std::forward<A>(a), std::forward<U>(u), alpha));
        return internal::make_fixed_square_adapter_like(std::move(tri));
      }
    }


#ifdef __cpp_concepts
    template<bool must_be_unique, bool must_be_exact, typename A, typename B> requires
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, A&&, B&&> or
      interface::solve_defined_for<NestedObject, must_be_unique, must_be_exact, nested_object_of_t<A&&>, B&&>
    static compatible_with_vector_space_descriptor_collection<std::tuple<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>>> auto
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

}


#endif
