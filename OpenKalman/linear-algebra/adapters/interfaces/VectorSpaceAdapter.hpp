/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Interfaces for VectorSpaceAdapter.
 */

#ifndef OPENKALMAN_INTERFACES_VECTORSPACEADAPTER_HPP
#define OPENKALMAN_INTERFACES_VECTORSPACEADAPTER_HPP


namespace OpenKalman::interface
{
  // --------------------------- //
  //   indexible_object_traits   //
  // --------------------------- //

  template<typename NestedObject, typename Descriptors>
  struct indexible_object_traits<VectorSpaceAdapter<NestedObject, Descriptors>>
  {
  private:

    using Xpr = VectorSpaceAdapter<NestedObject, Descriptors>;

  public:

    using scalar_type = scalar_type_of_t<NestedObject>;


    template<typename Arg>
    static constexpr auto count_indices(const Arg&)
    {
      // Truncate any trailing ℝ¹ dimensions
      using NewDesc = decltype(OpenKalman::internal::remove_trailing_1D_descriptors(std::declval<Descriptors>()));
      return OpenKalman::internal::collection_size_of<NewDesc>{};
    }


    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(Arg&& arg, const N& n)
    {
      constexpr auto dim = decltype(count_indices(arg))::value;

      if constexpr (euclidean_vector_space_descriptor_collection<Descriptors>)
      {
        return get_index_dimension_of(std::forward<Arg>(arg).nested_object(), n);
      }
      else if constexpr (vector_space_descriptor_tuple<Descriptors>)
      {
        static_assert (dim != dynamic_size);
        if constexpr (value::fixed<N>)
        {
          if constexpr (N::value >= dim)
            return descriptor::Axis{};
          else if constexpr (static_vector_space_descriptor<std::tuple_element_t<N::value, Descriptors>>)
            return std::tuple_element_t<N::value, Descriptors> {};
          else
            return std::get<N::value>(std::forward<Arg>(arg).my_descriptors);
        }
        else
        {
          return std::apply(
            [](auto&&...vs){ return std::array<descriptor::DynamicDescriptor<scalar_type>, dim> {std::forward<decltype(vs)>(vs)...}; },
            std::forward<Arg>(arg).my_descriptors)[n];
        }
      }
      else
      {
        using Dyn = descriptor::DynamicDescriptor<scalar_type>;
#ifdef __cpp_lib_ranges
        if (n >= std::ranges::size(arg.my_descriptors))
#else
        using std::size;
        if (n >= std::size(arg.my_descriptors))
#endif
        {
          return Dyn {descriptor::Axis{}};
        }
        else
        {
          return static_cast<Dyn>(std::forward<Arg>(arg).my_descriptors[n]);
        }
      }
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nested_object();
    }


    static constexpr auto get_constant(const Xpr& arg)
    {
      return constant_coefficient{arg.nested_object()};
    }


    static constexpr auto get_constant_diagonal(const Xpr& arg)
    {
      return constant_diagonal_coefficient {arg.nested_object()};
    }


    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedObject, b>;


    template<Qualification b>
    static constexpr bool is_square = square_shaped<NestedObject, b>;


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
  struct library_interface<VectorSpaceAdapter<Nested, Ds...>> : library_interface<std::decay_t<Nested>>
  {
  private:

    using NestedObject = std::decay_t<Nested>;
    using NestedInterface = library_interface<NestedObject>;

    template<typename T, typename Arg>
    static constexpr auto
    make_adapter(T&& t, Arg&& arg)
    {
      return std::apply([](auto&& a, auto&&...vs){
        return make_vector_space_adapter(std::forward<decltype(a)>(a), std::forward<decltype(vs)>(vs)...);
        }, std::tuple_cat(std::forward_as_tuple(std::forward<Arg>(arg)), std::forward<T>(t).my_descriptors));
    }

  public:

    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, NestedObject>;

#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires value::index<std::ranges::range_value_t<Indices>> and
      interface::get_component_defined_for<NestedObject, nested_object_of_t<Arg&&>, const Indices&>
    static constexpr value::scalar decltype(auto)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::get_component_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, const Indices&>, int> = 0>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      return NestedInterface::get_component(nested_object(std::forward<Arg>(arg)), indices);
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires value::index<std::ranges::range_value_t<Indices>> and
      interface::set_component_defined_for<NestedObject, nested_object_of_t<Arg&&>, const scalar_type_of_t<Arg>&, const Indices&>
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::set_component_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, const typename scalar_type_of<Arg>::type&, const Indices&>, int> = 0>
#endif
    static constexpr void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      NestedInterface::set_component(nested_object(arg), s, indices);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires interface::to_native_matrix_defined_for<NestedObject, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<interface::to_native_matrix_defined_for<NestedObject, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    static decltype(auto)
    to_native_matrix(Arg&& arg)
    {
      return make_adapter(std::forward<Arg>(arg), NestedInterface::to_native_matrix(nested_object(std::forward<Arg>(arg))));
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
    template<Layout layout, typename Scalar, typename D> requires
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&>
#else
    template<Layout layout, typename Scalar, typename D, std::enable_if_t<
      interface::make_default_defined_for<NestedObject, layout, Scalar, D&&>, int> = 0>
#endif
    static auto
    make_default(D&& d)
    {
      return NestedInterface::template make_default<layout, Scalar>(std::forward<D>(d));
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
      return make_adapter(std::forward<Arg>(arg), (NestedInterface::template make_triangular_matrix<t>(nested_object(std::forward<Arg>(arg)))));
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
      return make_adapter(std::forward<Arg>(arg), NestedInterface::template make_hermitian_adapter<t>(nested_object(std::forward<Arg>(arg))));
    }

  private:

    template<typename Arg, typename BeginTup, typename SizeTup, std::size_t...Ix>
    static decltype(auto)
    get_slice_impl(Arg&& arg, const BeginTup& begin_tup, const SizeTup& size_tup, std::index_sequence<Ix...>)
    {
      return make_vector_space_adapter(NestedInterface::get_slice(nested_object(std::forward<Arg>(arg)), begin_tup, size_tup),
        std::tuple {descriptor::get_slice<scalar_type_of_t<Arg>>(OpenKalman::get_vector_space_descriptor(arg, Ix), std::get<Ix>(begin_tup), std::get<Ix>(size_tup))...});
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
      return get_slice_impl(std::forward<Arg>(arg), begin_tup, size_tup, std::index_sequence_for<Begin...>{});
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
      return std::apply([](auto&& a, auto&& v, auto&&...vs){
        return make_vector_space_adapter(
          std::forward<decltype(a)>(a),
          std::forward<decltype(v)>(v),
          std::forward<decltype(v)>(v),
          std::forward<decltype(vs)>(vs)...);
        }, std::tuple_cat(
          std::forward_as_tuple(NestedInterface::to_diagonal(nested_object(std::forward<Arg>(arg)))),
          std::forward<Arg>(arg).my_descriptors),
          std::tuple{descriptor::Axis{}});
    }

  private:

    template<typename Arg, typename V0, typename V1, typename...Vs>
    static constexpr decltype(auto)
    diagonal_of_impl(Arg&& arg, V0&& v0, V1&& v1, const Vs&...vs)
    {
      auto d0 = internal::smallest_vector_space_descriptor<scalar_type_of_t<Arg>>(std::forward<V0>(v0), std::forward<V1>(v1));
      return make_vector_space_adapter(std::forward<Arg>(arg), d0, vs...);
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg> requires diagonal_adapter<NestedObject> or 
      (diagonal_adapter<NestedObject, 1> and 
        interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>) or 
      interface::diagonal_of_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr vector auto
#else
    template<typename Arg, std::enable_if_t<diagonal_adapter<NestedObject> or 
      (diagonal_adapter<NestedObject, 1> and 
        interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>) or 
      interface::diagonal_of_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    diagonal_of(Arg&& arg)
    {
      if constexpr (diagonal_adapter<NestedObject>)
        return diagonal_of_impl(nested_object(nested_object(std::forward<Arg>(arg))));
      else if constexpr (diagonal_adapter<NestedObject, 1> and 
          interface::transpose_defined_for<NestedObject, decltype(nested_object(nested_object(std::declval<Arg>())))>)
        return diagonal_of_impl(NestedInterface::transpose(nested_object(nested_object(std::forward<Arg>(arg)))));
      else 
        return diagonal_of_impl(NestedInterface::diagonal_of(nested_object(std::forward<Arg>(arg))),
          std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{descriptor::Axis{}, descriptor::Axis{}}));
    }

  private:

    template<std::size_t Ix, typename Arg, typename Factors_tup>
    static constexpr auto broadcast_for_index(const Arg& arg, const Factors_tup& factors_tup)
    {
      constexpr auto N = std::tuple_size_v<Factors_tup>;
      if constexpr (Ix < N)
        return get_vector_space_descriptor<Ix>(arg) * std::get<Ix>(factors_tup);
      else
        return descriptor::Axis{};
    }


    template<typename Arg, std::size_t...Is, typename Factors_tup>
    static constexpr auto broadcast_impl(Arg&& arg, std::index_sequence<Is...>, const Factors_tup& factors_tup)
    {
      constexpr auto N = std::tuple_size_v<Factors_tup>;
      return make_vector_space_adapter(std::forward<Arg>(arg), broadcast_for_index<Is>(arg, factors_tup)...);
    }

  public:

#ifdef __cpp_concepts
    template<indexible Arg, value::index...Factors> requires
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
    template<vector_space_descriptor...Vs, typename Operation> requires
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<Vs...>&, Operation&&>
    static indexible auto
#else
    template<typename...Vs, typename Operation, std::enable_if_t<
      interface::n_ary_operation_defined_for<NestedInterface, const std::tuple<Vs...>&, Operation&&>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<Vs...>& d_tup, Operation&& op)
    {
      return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op));
    }


#ifdef __cpp_concepts
    template<vector_space_descriptor...Vs, typename Operation, indexible Arg, indexible...Args> requires
      interface::n_ary_operation_defined_for<NestedObject, const std::tuple<Vs...>&, Operation&&, nested_object_of_t<Arg&&>, Args...>
    static indexible auto
#else
    template<typename...Vs, typename Operation, typename Arg, typename...Args, std::enable_if_t<
      interface::n_ary_operation_defined_for<NestedObject, const std::tuple<Vs...>&, Operation&&, typename nested_object_of<Arg&&>::type, Args...>, int> = 0>
    static auto
#endif
    n_ary_operation(const std::tuple<Vs...>& d_tup, Operation&& op, Arg&& arg, Args&&...args)
    {
      return NestedInterface::n_ary_operation(d_tup, std::forward<Operation>(op), nested_object(std::forward<Arg>(arg)), std::forward<Args>(args)...);
    }

  private:

    template<std::size_t...indices, typename Arg, typename...Vs, std::size_t...Ix>
    static constexpr decltype(auto)
    reduce_impl(Arg&& arg, const std::tuple<Vs...>& tup_vs, std::index_sequence<Ix...> seq)
    {
      return make_vector_space_adapter(std::forward<Arg>(arg),
        ([]{ constexpr auto I = Ix; return ((I == indices) or ...); } ?
          uniform_static_vector_space_descriptor_component_of_t<vector_space_descriptor_of_t<Vs, Ix>>{} :
          std::get<Ix>(tup_vs))...);
    }

  public:

#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, indexible Arg> requires
      interface::reduce_defined_for<NestedObject, BinaryFunction&&, nested_object_of_t<Arg&&>, indices...>
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
      interface::reduce_defined_for<NestedObject, BinaryFunction&&, typename nested_object_of<Arg&&>::type, indices...>, int> = 0>
#endif
    static constexpr auto
    reduce(BinaryFunction&& op, Arg&& arg)
    {
      return reduce_impl<indices...>(
        NestedInterface::template reduce<indices...>(std::forward<BinaryFunction>(op), nested_object(std::forward<Arg>(arg))),
        std::forward<Arg>(arg).my_descriptors,
        std::index_sequence_for<Ds...>{});
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::to_euclidean_defined_for<NestedObject, Arg&&> or
      interface::to_euclidean_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::to_euclidean_defined_for<NestedObject, Arg&&> or
      interface::to_euclidean_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    to_euclidean(Arg&& arg)
    {
      if constexpr (interface::to_euclidean_defined_for<NestedObject, Arg&&>)
        return NestedInterface::to_euclidean(std::forward<Arg>(arg));
      else
        return NestedInterface::to_euclidean(nested_object(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor V> requires
      interface::from_euclidean_defined_for<NestedObject, Arg&&, const V&> or
      interface::from_euclidean_defined_for<NestedObject, nested_object_of_t<Arg&&>, const V&>
    static constexpr indexible auto
#else
    template<typename Arg, typename V, std::enable_if_t<
      interface::from_euclidean_defined_for<NestedObject, Arg&&, const V&> or
      interface::from_euclidean_defined_for<NestedObject, typename nested_object_of<Arg&&>::type, const V&>, int> = 0>
    static constexpr auto
#endif
    from_euclidean(Arg&& arg, const V& v)
    {
      if constexpr (interface::from_euclidean_defined_for<NestedObject, Arg&&, const V&>)
        return NestedInterface::from_euclidean(std::forward<Arg>(arg), v);
      else
        return NestedInterface::from_euclidean(nested_object(std::forward<Arg>(arg)), v);
    }


#ifdef __cpp_concepts
    template<indexible Arg> requires
      interface::wrap_angles_defined_for<NestedObject, Arg&&> or
      interface::wrap_angles_defined_for<NestedObject, nested_object_of_t<Arg&&>>
    static constexpr indexible auto
#else
    template<typename Arg, std::enable_if_t<
      interface::wrap_angles_defined_for<NestedObject, Arg&&> or
      interface::wrap_angles_defined_for<NestedObject, typename nested_object_of<Arg&&>::type>, int> = 0>
    static constexpr auto
#endif
    wrap_angles(Arg&& arg)
    {
      if constexpr (interface::wrap_angles_defined_for<NestedObject, Arg&&>)
        return NestedInterface::wrap_angles(std::forward<Arg>(arg));
      else
        return NestedInterface::wrap_angles(nested_object(std::forward<Arg>(arg)));
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

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INTERFACES_VECTORSPACEADAPTER_HPP
