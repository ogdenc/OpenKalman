/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief ToEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_TOEUCLIDEANEXPR_HPP
#define OPENKALMAN_TOEUCLIDEANEXPR_HPP

#include "basics/traits/traits.hpp"

namespace OpenKalman
{

  /// \todo Remove nested diagonal matrix option
#ifdef __cpp_concepts
  template<indexible NestedObject>
#else
  template<typename NestedObject>
#endif
  struct ToEuclideanExpr : internal::AdapterBase<ToEuclideanExpr<NestedObject>, NestedObject>
  {

  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
#endif

    using Scalar = scalar_type_of_t<NestedObject>;

    using Base = internal::AdapterBase<ToEuclideanExpr, NestedObject>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr ToEuclideanExpr() requires std::default_initializable<Base>
#else
    template<typename B = Base, std::enable_if_t<std::is_default_constructible_v<B>, int> = 0>
    constexpr ToEuclideanExpr()
#endif
    {}


    /**
     * \brief Construct from compatible \ref indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires std::constructible_from<NestedObject, Arg&&>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and std::is_constructible_v<NestedObject, Arg&&>, int> = 0>
#endif
    explicit ToEuclideanExpr(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a compatible \ref indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>,
        decltype(from_euclidean(std::declval<Arg>(), get_vector_space_descriptor<0>(std::declval<NestedObject>())))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>,
        decltype(from_euclidean(std::declval<Arg>(), get_vector_space_descriptor<0>(std::declval<NestedObject>())))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      using FArg = decltype(from_euclidean(std::declval<Arg>(), get_vector_space_descriptor<0>(std::declval<NestedObject>())));
      if constexpr ((zero<NestedObject> and zero<FArg>) or (identity_matrix<NestedObject> and identity_matrix<FArg>))
      {}
      else
      {
        this->nested_object() = from_euclidean(std::forward<Arg>(arg), get_vector_space_descriptor<0>(nested_object(arg)));
      }
      return *this;
    }

  };


  // ------------------------------ //
  //        Deduction Guide         //
  // ------------------------------ //

#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  ToEuclideanExpr(Arg&&) -> ToEuclideanExpr<Arg>;


  namespace interface
  {

    // --------------------------- //
    //   indexible_object_traits   //
    // --------------------------- //

    template<typename NestedObject>
    struct indexible_object_traits<ToEuclideanExpr<NestedObject>>
    {
      using scalar_type = scalar_type_of_t<NestedObject>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }


      template<typename Arg, typename N>
      static constexpr auto
      get_vector_space_descriptor(Arg&& arg, const N& n)
      {
        if constexpr (value::fixed<N>)
        {
          if constexpr (n == 0_uz) return Axis;
          else return OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Desc = DynamicDescriptor<scalar_type_of<Arg>>;
          if (n == 0) return Desc {Axis};
          else return Desc {OpenKalman::get_vector_space_descriptor(nested_object(std::forward<Arg>(arg)), n)};
        }
      }


      template<typename Arg>
      static decltype(auto)
      nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto
      get_constant(const Arg& arg)
      {
        if constexpr (has_untyped_index<NestedObject, 0>)
          return constant_coefficient {arg.nested_object()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto
      get_constant_diagonal(const Arg& arg)
      {
        if constexpr (has_untyped_index<NestedObject, 0>)
          return constant_diagonal_coefficient {arg.nested_object()};
        else
          return std::monostate {};
      }


      template<Qualification b>
      static constexpr bool
      one_dimensional = has_untyped_index<NestedObject, 0> and OpenKalman::one_dimensional<NestedObject, b>;


      template<Qualification b>
      static constexpr bool
      is_square = has_untyped_index<NestedObject, 0> and square_shaped<NestedObject, b>;


      template<TriangleType t>
      static constexpr bool
      is_triangular = has_untyped_index<NestedObject, 0> and triangular_matrix<NestedObject, t>;


      static constexpr bool
      is_triangular_adapter = false;


      static constexpr bool
      is_hermitian = has_untyped_index<NestedObject, 0> and hermitian_matrix<NestedObject>;


      // hermitian_adapter_type is omitted


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires has_untyped_index<NestedObject, 0> and raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<has_untyped_index<NestedObject, 0> and raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg)
      {
        return internal::raw_data(OpenKalman::nested_object(arg));
      }


      static constexpr Layout
      layout = has_untyped_index<NestedObject, 0> ? layout_of_v<NestedObject> : Layout::none;


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


    // --------------------- //
    //   library_interface   //
    // --------------------- //

    template<typename NestedObject>
    struct library_interface<ToEuclideanExpr<NestedObject>>
    {
    private:

        using NestedInterface = library_interface<NestedObject>;

    public:

      template<typename Derived>
      using LibraryBase = internal::library_base_t<Derived, pattern_matrix_of_t<T>>;


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires value::index<std::ranges::range_value_t<Indices>>
      static constexpr value::scalar decltype(auto)
#else
      template<typename Arg, typename Indices>
      static constexpr decltype(auto)
#endif
      get_component(Arg&& arg, const Indices& indices)
      {
        if constexpr (has_untyped_index<NestedObject, 0>)
        {
          return NestedInterface::get_component(nested_object(std::forward<Arg>(arg)), indices);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return get_component(nested_object(std::forward<Arg>(arg)), ix, is...); }};
          return to_euclidean_element(get_vector_space_descriptor<0>(arg), g, i, 0);
        }
      }


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires value::index<std::ranges::range_value_t<Indices>>
#else
      template<typename Arg, typename Indices>
#endif
      static void
      set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
      {
        if constexpr (has_untyped_index<NestedObject, 0>)
        {
          NestedInterface::set_component(nested_object(arg), s, indices);
        }
        else
        {
          set_component(nested_object(arg), s, indices);
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<nested_object_of_t<Arg>>(std::forward<Arg>(arg));
      }


      template<Layout layout, typename Scalar, typename D>
      static auto make_default(D&& d)
      {
        return make_dense_object<NestedObject, layout, Scalar>(std::forward<D>(d));
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename D>
      static constexpr auto make_constant(C&& c, D&& d)
      {
        return make_constant<NestedObject>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename Scalar, typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<NestedObject, Scalar>(std::forward<D>(d));
      }


      // get_slice


      // set_slice


      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
        if constexpr( has_untyped_index<NestedObject, 0>)
        {
          return to_diagonal(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          using P = pattern_matrix_of_t<T>;
          return library_interface<P>::to_diagonal(to_native_matrix<P>(std::forward<Arg>(arg)));
        }
      }


      template<typename Arg>
      static auto
      diagonal_of(Arg&& arg)
      {
        if constexpr(has_untyped_index<NestedObject, 0>)
        {
          return diagonal_of(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          using P = pattern_matrix_of_t<T>;
          return library_interface<P>::diagonal_of(to_native_matrix<P>(std::forward<Arg>(arg)));
        }
      }


      template<typename Arg, typename...Factors>
      static auto
      broadcast(Arg&& arg, const Factors&...factors)
      {
        return library_interface<std::decay_t<nested_object_of_t<Arg>>>::broadcast(std::forward<Arg>(arg), factors...);
      }


      template<typename...Ds, typename Operation, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
      {
        using P = pattern_matrix_of_t<T>;
        return library_interface<P>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        using P = pattern_matrix_of_t<T>;
        return library_interface<P>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      // to_euclidean not included

      // from_duclidean not included

      // wrap_angles not included


      template<typename Arg>
      static constexpr decltype(auto)
      conjugate(Arg&& arg)
      {
        if constexpr(has_untyped_index<NestedObject, 0>)
        {
          return OpenKalman::conjugate(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).conjugate(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr decltype(auto)
      transpose(Arg&& arg)
      {
        if constexpr(has_untyped_index<NestedObject, 0>)
        {
          return OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).transpose(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr decltype(auto)
      adjoint(Arg&& arg)
      {
        if constexpr(has_untyped_index<NestedObject, 0>)
        {
          return OpenKalman::adjoint(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).adjoint(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr auto
      determinant(Arg&& arg)
      {
        if constexpr(has_untyped_index<NestedObject, 0>)
        {
          return OpenKalman::determinant(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return arg.determinant(); //< \todo Generalize this.
        }
      }


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto)
      rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_hermitian<significant_triangle>(make_hermitian_matrix(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
      }


      template<TriangleType triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_triangular(make_triangular_matrix<triangle>(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
      }


      template<bool must_be_unique, bool must_be_exact, typename A, typename B>
      static constexpr decltype(auto)
      solve(A&& a, B&& b)
      {
        return OpenKalman::solve<must_be_unique, must_be_exact>(
          to_native_matrix<T>(std::forward<A>(a)), std::forward<B>(b));
      }


      template<typename A>
      static inline auto
      LQ_decomposition(A&& a)
      {
        return LQ_decomposition(to_dense_object(std::forward<A>(a)));
      }


      template<typename A>
      static inline auto
      QR_decomposition(A&& a)
      {
        return QR_decomposition(to_dense_object(std::forward<A>(a)));
      }

    };


  } // namespace interface


} // OpenKalman



#endif //OPENKALMAN_TOEUCLIDEANEXPR_HPP
