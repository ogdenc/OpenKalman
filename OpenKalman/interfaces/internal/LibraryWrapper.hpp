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
 * \brief Interfaces for LibraryWrapper
 */

#ifndef OPENKALMAN_INTERFACES_LIBRARYWRAPPER_HPP
#define OPENKALMAN_INTERFACES_LIBRARYWRAPPER_HPP

namespace OpenKalman::interface
{
  // ------------------------- //
  //  indexible_object_traits  //
  // ------------------------- //

  template<typename NestedObject, typename LibraryObject>
  struct indexible_object_traits<internal::LibraryWrapper<NestedObject, LibraryObject>>
  {
    using scalar_type = scalar_type_of_t<NestedObject>;


    template<typename Arg>
    static constexpr auto get_index_count(const Arg& arg) { return OpenKalman::get_index_count(arg.nested_matrix()); }


    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nested_matrix(), n);
    }


    using dependents = std::tuple<NestedObject>;


    static constexpr bool has_runtime_parameters = false;


    template<std::size_t i, typename Arg>
    static constexpr decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nested_matrix();
    }


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      return make_dense_writable_matrix_from(OpenKalman::nested_matrix(std::forward<Arg>(arg)));
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient{arg.nested_matrix()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {arg.nested_matrix()};
    }


    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<NestedObject, b>;


    template<Likelihood b>
    static constexpr bool is_square = square_matrix<NestedObject, b>;


    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<NestedObject, t, b>;


    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<NestedObject, Likelihood::maybe>;


    static constexpr bool is_writable = writable<NestedObject>;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&>>
#else
    template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
    static constexpr auto*
    data(Arg& arg) { return internal::raw_data(arg.nested_matrix()); }


    static constexpr Layout layout = layout_of_v<NestedObject>;


#ifdef __cpp_concepts
    template<typename Arg> requires (layout != Layout::none)
#else
    template<Layout l = layout, typename Arg, std::enable_if_t<l != Layout::none, int> = 0>
#endif
    static auto
    strides(Arg&& arg)
    {
      return OpenKalman::internal::strides(std::forward<Arg>(arg));
    }

  };


  // ------------------- //
  //  library_interface  //
  // ------------------- //

  template<typename NestedObject, typename LibraryObject>
  struct library_interface<internal::LibraryWrapper<NestedObject, LibraryObject>>
    : library_interface<std::decay_t<LibraryObject>>
  {
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (interface::get_component_defined_for<std::decay_t<NestedObject>, decltype(nested_matrix(std::declval<Arg&&>())), const Indices&> or
        interface::get_component_defined_for<std::decay_t<LibraryObject>, decltype(std::declval<Arg&&>()), const Indices&>)
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::get_component_defined_for<std::decay_t<NestedObject>, decltype(nested_matrix(std::declval<Arg&&>())), const Indices&> or
      interface::get_component_defined_for<std::decay_t<LibraryObject>, decltype(std::declval<Arg&&>()), const Indices&>, int> = 0>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices)
    {
      if constexpr (interface::get_component_defined_for<NestedObject, decltype(nested_matrix(std::declval<Arg&&>())), const Indices&>)
        return library_interface<std::decay_t<NestedObject>>::get_component(nested_matrix(std::forward<Arg>(arg)), indices);
      else
        return library_interface<std::decay_t<LibraryObject>>::get_component(std::forward<Arg>(arg), indices);
    }


#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (interface::set_component_defined_for<std::decay_t<NestedObject>, decltype(nested_matrix(std::declval<Arg&>())), const scalar_type_of_t<Arg>&, const Indices&> or
        interface::set_component_defined_for<std::decay_t<LibraryObject>, decltype(std::declval<Arg&>()), const scalar_type_of_t<Arg>&, const Indices&>)
#else
    template<typename Arg, typename Indices, std::enable_if_t<
      interface::set_component_defined_for<std::decay_t<NestedObject>, decltype(nested_matrix(std::declval<Arg&>())), const typename scalar_type_of<Arg>::type&, const Indices&> or
      interface::set_component_defined_for<std::decay_t<LibraryObject>, decltype(std::declval<Arg&>()), const typename scalar_type_of<Arg>::type&, const Indices&>, int> = 0>
#endif
    static constexpr void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
    {
      if constexpr (interface::set_component_defined_for<NestedObject, decltype(nested_matrix(std::declval<Arg&>())), const scalar_type_of_t<Arg>&, const Indices&>)
        library_interface<std::decay_t<NestedObject>>::set_component(nested_matrix(arg), s, indices);
      else
        library_interface<std::decay_t<LibraryObject>>::set_component(arg, s, indices);
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_INTERFACES_LIBRARYWRAPPER_HPP