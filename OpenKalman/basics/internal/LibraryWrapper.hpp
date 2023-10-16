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
 * \brief Definitions for LibraryWrapper
 * \todo Possibly combine with FixedSizeAdapter.
 */

#ifndef OPENKALMAN_LIBRARYWRAPPER_HPP
#define OPENKALMAN_LIBRARYWRAPPER_HPP

namespace OpenKalman
{
  namespace internal
  {
    /**
     * \internal
     * \brief A dumb wrapper for \ref indexible objects so that they are treated exactly as native objects within a library.
     * \tparam NestedObject An indexible object that may or may not be in a library of interest.
     * \tparam LibraryObject Any object from the library to which this wrapper is to be associated.
     */
#ifdef __cpp_concepts
    template<indexible NestedObject, indexible LibraryObject = NestedObject>
#else
    template<typename NestedObject, typename LibraryObject = NestedObject>
#endif
    struct LibraryWrapper : internal::library_base_t<LibraryWrapper<NestedObject, LibraryObject>, LibraryObject>
    {
    private:

      using Base = internal::library_base_t<LibraryWrapper, LibraryObject>;

    public:

#ifdef __cpp_concepts
      template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
      std::constructible_from<NestedObject, Arg&&>
#else
      template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>) and
        std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
      explicit LibraryWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}


#ifdef __cpp_concepts
      template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, LibraryWrapper>) and
      (not std::constructible_from<NestedObject, Arg&&>) and std::default_initializable<NestedObject> and
        element_settable<NestedObject, 2>
#else
      template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, LibraryWrapper>) and
      (not std::is_constructible_v<NestedObject, Arg&&>) and std::is_default_constructible_v<NestedObject> and
      element_settable<NestedObject, 2>, int> = 0>
#endif
      explicit LibraryWrapper(Arg&& arg)
      {
        for (std::size_t i = 0; i < get_index_dimension_of<0>(wrapped_expression); i++)
        for (std::size_t j = 0; j < get_index_dimension_of<1>(wrapped_expression); j++)
          set_element(wrapped_expression, get_element(std::forward<Arg>(arg), i, j), i, j);
      }


      /**
       * \brief Assign from another compatible indexible object.
       */
  #ifdef __cpp_concepts
      template<indexible Arg> requires
        std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>
  #else
      template<typename Arg, std::enable_if_t<
        std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_native_matrix<NestedObject>(std::declval<Arg&&>()))>, int> = 0>
  #endif
      auto& operator=(Arg&& arg) noexcept
      {
        wrapped_expression = to_native_matrix<NestedObject>(std::forward<Arg>(arg));
        return *this;
      }


      /**
       * \brief Get the nested matrix.
       */
      decltype(auto) nested_matrix() & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() && noexcept { return (std::move(*this).wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const && noexcept { return (std::move(*this).wrapped_expression); }

    protected:

      NestedObject wrapped_expression;

    };


  } // namespace internal


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename NestedObject, typename LibraryObject>
    struct indexible_object_traits<internal::LibraryWrapper<NestedObject, LibraryObject>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;

      template<typename Arg>
      static constexpr auto get_index_count(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        return OpenKalman::get_vector_space_descriptor(nested_matrix(arg), n);
      }

      using type = std::tuple<NestedObject>;

      static constexpr bool has_runtime_parameters = false;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg).nested_matrix());
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

#ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<decltype(nested_matrix(std::declval<Arg&&>())), sizeof...(I)>
#else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<Arg&&>())), sizeof...(I)>, int> = 0>
#endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename Scalar, typename...I> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), sizeof...(I)>
#else
      template<typename Arg, typename Scalar, typename...I, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), sizeof...(I)>, int> = 0>
#endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
      {
        set_element(nested_matrix(arg), s, i...);
      }


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


    template<typename NestedObject, typename LibraryObject>
    struct library_interface<internal::LibraryWrapper<NestedObject, LibraryObject>> : library_interface<LibraryObject>
    {
      template<typename Derived>
      using LibraryBase = internal::library_base_t<Derived, NestedObject>;
    };

  } // namespace interface


} // namespace OpenKalman

#endif //OPENKALMAN_EIGENWRAPP