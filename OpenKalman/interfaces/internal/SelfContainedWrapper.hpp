/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Interfaces for \ref SelfContainedWrapper
 */

#ifndef OPENKALMAN_INTERFACES_SELFCONTAINEDWRAPPER_HPP
#define OPENKALMAN_INTERFACES_SELFCONTAINEDWRAPPER_HPP

namespace OpenKalman::interface
{
  // ------------------------- //
  //  indexible_object_traits  //
  // ------------------------- //

  template<typename NestedObject, typename...InternalizedParameters>
  struct indexible_object_traits<internal::SelfContainedWrapper<NestedObject, InternalizedParameters...>>
  {
    using scalar_type = scalar_type_of_t<NestedObject>;


    template<typename Arg>
    static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(arg.nested_object()); }


    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nested_object(), n);
    }


    using dependents = std::tuple<NestedObject, InternalizedParameters...>;


    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static constexpr decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nested_object();
    }


    template<typename Arg>
    static decltype(auto) convert_to_self_contained(Arg&& arg)
    {
      if constexpr (std::is_lvalue_reference_v<NestedObject>)
        return make_dense_object(OpenKalman::nested_object(std::forward<Arg>(arg)));
      else
        return std::forward<Arg>(arg);
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient{arg.nested_object()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
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
    template<typename Arg> requires directly_accessible<nested_object_of_t<Arg&>>
#else
    template<typename Arg, std::enable_if_t<directly_accessible<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
    static constexpr auto*
    raw_data(Arg& arg) { return internal::raw_data(arg.nested_object()); }


    static constexpr Layout layout = layout_of_v<NestedObject>;


#ifdef __cpp_concepts
    template<typename Arg> requires (layout != Layout::none)
#else
    template<Layout l = layout, typename Arg, std::enable_if_t<l != Layout::none, int> = 0>
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

  template<typename NestedObject, typename...InternalizedParameters>
  struct library_interface<internal::SelfContainedWrapper<NestedObject, InternalizedParameters...>>
    : library_interface<std::decay_t<NestedObject>>
  {

#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      interface::get_component_defined_for<std::decay_t<NestedObject>, decltype(nested_object(std::declval<Arg&&>())), const Indices&>
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::get_component_defined_for<std::decay_t<NestedObject>, decltype(nested_object(std::declval<Arg&&>())), const Indices&>, int> = 0>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      return library_interface<std::decay_t<NestedObject>>::get_component(nested_object(std::forward<Arg>(arg)), indices);
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      interface::set_component_defined_for<std::decay_t<NestedObject>, decltype(nested_object(std::declval<Arg&>())), const scalar_type_of_t<Arg>&, const Indices&>
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::set_component_defined_for<std::decay_t<NestedObject>, decltype(nested_object(std::declval<Arg&>())), const typename scalar_type_of<Arg>::type&, const Indices&>, int> = 0>
#endif
    static constexpr void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      library_interface<std::decay_t<NestedObject>>::set_component(nested_object(arg), s, indices);
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_INTERFACES_SELFCONTAINEDWRAPPER_HPP